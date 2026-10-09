"""Independent A28 accounting, adapted from the frozen A27 implementation.

Only results/a28/jobs restores A28 history. A26 and A25 historical receipts are never
edited or charged again. History is scanned once when a charged job starts;
operator checks use in-memory prior totals and owned-lock checks.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from pathlib import Path
import json
import math
import os
import time
import uuid

from a20.costs import CostBook, BudgetExceeded, plain

CPU_LIMIT_SECONDS = 3600.0
GPU_LIMIT_SECONDS = 3600.0
MAX_NEW_LABELS = 0
MAX_FULL_WAVE_MATERIAL_STATES = 768
SHARED_GPU_LOCK = Path("D:/AI/A20_OPM_IMAGING/runs/gpu.lock")
TERMINAL_STATUSES = frozenset({"COMPLETE", "FAILED", "BUDGET_STOP", "INTERRUPTED"})
EVIDENCE_ONLY_STAGES = frozenset({"report", "evidence-report", "accounting", "delivery-preparation",
                                  "public-delivery", "transport"})


def _evidence_only(device, metadata):
    """After a state-cap violation, only named CPU evidence work may start."""
    return device == "cpu" and (metadata or {}).get("stage") in EVIDENCE_ONLY_STAGES


def _category(category):
    category = str(category)
    if category in {"new_labels", "new_clean_labels"}:
        return "new_full_forward_labels"
    if category in {"full_wave_states", "material_state_evaluations", "full_wave_material_state"}:
        return "full_wave_material_states"
    return category


def _state_cap(value):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_FULL_WAVE_MATERIAL_STATES:
        raise ValueError("A28_FULL_WAVE_STATE_CAP_CANNOT_BE_EXPANDED")
    return value


def write_json(path, value):
    """Atomic, flushed metadata replacement (never used to replace a receipt)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temp.open("x", encoding="utf-8") as handle:
        json.dump(plain(value), handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _number(row, key):
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise BudgetExceeded("INVALID_ACCEPTED_RESOURCE_RECEIPT:" + key)
    return float(value)


def budget_history(root):
    """Restore only A28 attempts, including failures and reservations.

    A dangling started manifest blocks new work: its unknown interruption tail
    cannot be silently replaced by zero or a last checkpoint.
    """
    jobs = Path(root) / "results/a28/jobs"
    total_cpu = total_gpu = 0.0
    reservations = Counter()
    receipts, unresolved, legacy_fixtures = [], [], []
    unreserved_fixture_states = 0
    for directory in sorted(jobs.glob("*")):
        if not directory.is_dir():
            continue
        manifest, receipt = directory / "job_manifest.json", directory / "job_receipt.json"
        if not receipt.exists():
            if manifest.exists():
                unresolved.append(directory.name)
            continue
        try:
            row = json.loads(receipt.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise BudgetExceeded("UNREADABLE_ACCEPTED_RECEIPT:" + directory.name) from exc
        if row.get("status") not in TERMINAL_STATUSES or row.get("job") != directory.name:
            raise BudgetExceeded("NONTERMINAL_OR_MISMATCHED_RECEIPT:" + directory.name)
        wall, process = _number(row, "wall_seconds"), _number(row, "process_cpu_seconds")
        device = row.get("device")
        if device not in {"cpu", "cuda"}:
            raise BudgetExceeded("INVALID_RECEIPT_DEVICE")
        cpu = max(wall, process) if device == "cpu" else process
        gpu = _number(row, "gpu_occupation_seconds") if device == "cuda" else 0.0
        host_only = row.get("gpu_host_only", False)
        if not isinstance(host_only, bool) or (host_only and device != "cuda"):
            raise BudgetExceeded("INVALID_RECEIPT_GPU_HOST_FLAG")
        expected_gpu = wall if device == "cuda" and not host_only else 0.0
        if abs(gpu - expected_gpu) > 1e-6:
            raise BudgetExceeded("RECEIPT_GPU_OCCUPATION_CHARGE_MISMATCH")
        if "charged_cpu_seconds" in row and abs(_number(row, "charged_cpu_seconds") - cpu) > 1e-6:
            raise BudgetExceeded("RECEIPT_CPU_CHARGE_MISMATCH")
        saved_reservations = row.get("durable_reservations", {})
        if not isinstance(saved_reservations, dict) or any(
                isinstance(v, bool) or not isinstance(v, int) or v < 0
                for v in saved_reservations.values()):
            raise BudgetExceeded("INVALID_RESERVATION_RECEIPT")
        local = Counter({str(k): v for k, v in saved_reservations.items()})
        # Durable action reservations survive a receipt written just before a
        # flush boundary. Never sum the same local reservation twice.
        ledger = directory / "COST_LEDGER.jsonl"
        if ledger.exists():
            with ledger.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        action = json.loads(line)
                    except ValueError as exc:
                        raise BudgetExceeded("TRUNCATED_DURABLE_LEDGER:" + directory.name) from exc
                    if action.get("event") == "RESERVED":
                        category, used = str(action.get("category", "")), action.get("job_used")
                        if not category or isinstance(used, bool) or not isinstance(used, int) or used < 0:
                            raise BudgetExceeded("INVALID_DURABLE_RESERVATION:" + directory.name)
                        local[category] = max(local[category], used)
        for category, count in local.items():
            if _category(category) == "new_full_forward_labels" and count:
                raise BudgetExceeded("A28_ZERO_NEW_LABEL_CAP_VIOLATED:" + directory.name)
            reservations[_category(category)] += count
        # Historical inherited real Physics2D(8) tests bypassed CountedMaxwell.
        # They use geometric Green operators, so their states count at every
        # resolution. Keep old receipts unchanged and expose this supplement.
        # Non-test audit full_forward_calls were already manually reserved.
        if row.get("stage") == "tests":
            calls = row.get("counts", {}).get("full_forward_calls", 0)
            central = row.get("central_full_forward_span_reservations", 0)
            if any(isinstance(v, bool) or not isinstance(v, int) or v < 0
                   for v in (calls, central)) or central > calls:
                raise BudgetExceeded("INVALID_FULL_FORWARD_SPAN_RECEIPT:" + directory.name)
            missing = calls - central
            if missing:
                legacy_fixtures.append({"job": directory.name,
                    "full_forward_calls": calls, "centrally_reserved_span_calls": central,
                    "legacy_unreserved_real_fixture_states": missing,
                    "resolution": "8x8 actual geometric A2 TM backend",
                    "receipt_unchanged": True})
                unreserved_fixture_states += missing
        total_cpu += cpu
        total_gpu += gpu
        receipts.append(row)
    campaign_states = reservations["full_wave_material_states"]
    strict_states = campaign_states + unreserved_fixture_states
    reservations["full_wave_material_states"] = strict_states
    # Native dense-reference correctness solves bypassed this counter too.
    # Preserve their raw operator/RHS counters; do not fabricate an exact total
    # of additional material queries or count repeated RHS as new states.
    dense_oracle_audits = sum(row.get("counts", {}).get("evaluator_dense_oracles", 0)
                             for row in receipts if row.get("stage") == "tests")
    return {"cpu_seconds": total_cpu, "gpu_seconds": total_gpu,
            "reservations": dict(reservations), "receipts": receipts,
            "unresolved_jobs": unresolved,
            "full_wave_material_states": strict_states,
            "campaign_full_wave_material_states": campaign_states,
            "legacy_unreserved_real_fixture_states": unreserved_fixture_states,
            "legacy_real_fixture_jobs": legacy_fixtures,
            "strict_all_resolution_full_wave_material_states": strict_states,
            "strict_state_total_is_known_lower_bound": bool(legacy_fixtures or dense_oracle_audits),
            "strict_state_count_label": "AT_LEAST" if legacy_fixtures or dense_oracle_audits else "EXACT_CENTRAL_RESERVATIONS",
            "unclassified_legacy_dense_oracle_audits": dense_oracle_audits,
            "full_wave_material_state_cap": MAX_FULL_WAVE_MATERIAL_STATES,
            "full_wave_material_state_budget_compliance":
                "FAIL" if strict_states > MAX_FULL_WAVE_MATERIAL_STATES else
                "NOT_ESTABLISHED" if legacy_fixtures or dense_oracle_audits else "PASS",
            "full_wave_material_states_over_cap": max(0, strict_states - MAX_FULL_WAVE_MATERIAL_STATES)}


@contextmanager
def owned_file_lock(path, *, owner_id, job):
    """Compatible with the A20 shared JSON file lock; release only our bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"pid": os.getpid(), "started_utc":
                         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                         "owner_id": str(owner_id), "job": str(job)}, sort_keys=True).encode()
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise BudgetExceeded("EXISTING_JOB_LOCK; inspect owner, never silently delete") from exc
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        yield (path, payload)
    finally:
        # A changed/foreign lock is retained for inspection, never deleted.
        if path.exists() and path.read_bytes() == payload:
            path.unlink()


class BudgetLedger(CostBook):
    """CostBook-compatible spans plus inclusive cumulative A28 limits."""
    def __init__(self, path=None, *, device="cpu", prior_cpu=0.0, prior_gpu=0.0,
                 prior_reservations=None, cpu_limit=CPU_LIMIT_SECONDS,
                 gpu_limit=GPU_LIMIT_SECONDS, cpu_reserve=15.0, gpu_reserve=30.0,
                 metadata=None, checkpoint_path=None, enforce=True,
                 started_wall=None, started_cpu=None, gpu_host_only=False,
                 prior_campaign_states=None, prior_legacy_fixture_states=0):
        if device not in {"cpu", "cuda"}:
            raise ValueError("A28_CPU_OR_CUDA_REQUIRED")
        if not 0 < float(cpu_limit) <= CPU_LIMIT_SECONDS or not 0 < float(gpu_limit) <= GPU_LIMIT_SECONDS:
            raise ValueError("A28_BUDGET_CANNOT_BE_EXPANDED")
        for name, value in (("prior_cpu", prior_cpu), ("prior_gpu", prior_gpu),
                            ("cpu_reserve", cpu_reserve), ("gpu_reserve", gpu_reserve)):
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError("INVALID_A28_BUDGET_PARAMETER:" + name)
        if float(cpu_reserve) >= float(cpu_limit) or float(gpu_reserve) >= float(gpu_limit):
            raise ValueError("A28_STOP_RESERVE_MUST_FIT_LIMIT")
        self.accounting_device = device
        self.gpu_host_only = bool(gpu_host_only)
        if self.gpu_host_only and device != "cuda":
            raise ValueError("GPU_HOST_OVERHEAD_REQUIRES_CUDA_ACCOUNTING")
        super().__init__(path, device="cpu" if self.gpu_host_only else device, prior_cpu=prior_cpu, prior_gpu=prior_gpu,
                         cpu_limit=cpu_limit, gpu_limit=gpu_limit,
                         cpu_reserve=cpu_reserve, gpu_reserve=gpu_reserve,
                         metadata=metadata, enforce=False)
        if started_wall is not None:
            self.started_wall = float(started_wall)
        if started_cpu is not None:
            self.started_cpu = float(started_cpu)
        self._budget_enforce = bool(enforce)
        self.prior_reservations = Counter()
        for category, count in (prior_reservations or {}).items():
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("INVALID_A28_PRIOR_RESERVATION")
            if _category(category) == "new_full_forward_labels" and count:
                raise BudgetExceeded("A28_ZERO_NEW_LABEL_CAP_VIOLATED")
            self.prior_reservations[_category(category)] += count
        self.reservations = Counter()
        self.central_full_forward_span_reservations = 0
        self.full_wave_limit = MAX_FULL_WAVE_MATERIAL_STATES
        self.evidence_only = _evidence_only(device, self.metadata)
        self.prior_campaign_states = (self.prior_reservations["full_wave_material_states"]
                                      if prior_campaign_states is None else prior_campaign_states)
        self.prior_legacy_fixture_states = prior_legacy_fixture_states
        if (self.prior_reservations["full_wave_material_states"] > self.full_wave_limit
                and not self.evidence_only):
            raise BudgetExceeded("A28_FULL_WAVE_STATE_CAP_ALREADY_VIOLATED")
        self.checkpoint_path = None if checkpoint_path is None else Path(checkpoint_path)
        self._last_checkpoint = -float("inf")
        self._last_resource_sample_wall = -float("inf")
        self._owned_locks = []
        self.owner_id = self.metadata.get("owner_id")
        self.deadline = self.started_wall + max(0.0, self.cpu_limit - self.cpu_reserve - self.prior_cpu)
        if device == "cuda" and not self.gpu_host_only:
            self.deadline = self.started_wall + max(0.0, self.gpu_limit - self.gpu_reserve - self.prior_gpu)
        elif self.gpu_host_only:
            self.deadline = float("inf")

    def check_owner(self, owner_id):
        if self.owner_id is not None and str(owner_id) != str(self.owner_id):
            raise BudgetExceeded("FOREIGN_BUDGET_OWNER")
        for path, payload in self._owned_locks:
            if not path.exists() or path.read_bytes() != payload:
                raise BudgetExceeded("JOB_LOCK_OWNERSHIP_LOST")

    def elapsed(self):
        wall = max(0.0, time.perf_counter() - self.started_wall)
        process = max(0.0, time.process_time() - self.started_cpu)
        return wall, process, max(wall, process) if self.accounting_device == "cpu" else process

    def check(self):
        self.check_owner(self.owner_id)
        wall, process, cpu = self.elapsed()
        if self.checkpoint_path is not None and wall - self._last_checkpoint >= 1.0:
            write_json(self.checkpoint_path, self.receipt())
            self._last_checkpoint = wall
        if self._budget_enforce:
            if self.prior_cpu + cpu >= self.cpu_limit - self.cpu_reserve:
                raise BudgetExceeded("A28_CUMULATIVE_CPU_LIMIT")
            if self.accounting_device == "cuda" and not self.gpu_host_only and self.prior_gpu + wall >= self.gpu_limit - self.gpu_reserve:
                raise BudgetExceeded("A28_CUMULATIVE_GPU_OCCUPATION_LIMIT")
        # The inherited memory checks remain active; inherited CPU-only
        # process accounting is disabled in favor of the rule above.
        if wall - self._last_resource_sample_wall >= 1.0:
            super().check()
            self._last_resource_sample_wall = wall

    def append(self, row):
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(plain(row), allow_nan=False) + "\n")
                handle.flush()
                if row.get("event") == "RESERVED":
                    os.fsync(handle.fileno())
        self.events += 1

    def reserve(self, category, limit=None):
        """Reserve before an attempted action, retaining charge after failure.

        Cached labels do not call this. Every new exact material state calls
        reserve('full_wave_material_states') *before* factorization/solving.
        A caller cannot enlarge the global cap by supplying a larger limit.
        """
        self.check()
        category = _category(category)
        if (self.prior_reservations["full_wave_material_states"] > self.full_wave_limit
                or self.evidence_only):
            self.append({"event": "PHYSICS_RESERVATION_DENIED", "category": category,
                         "strict_prior_states": self.prior_reservations["full_wave_material_states"],
                         "status": "DENIED", **self.metadata})
            raise BudgetExceeded("A28_EVIDENCE_ONLY_NO_NEW_PHYSICS")
        if category == "new_full_forward_labels":
            self.append({"event": "LABEL_RESERVATION_DENIED", "category": category,
                         "limit": 0, "status": "DENIED", **self.metadata})
            raise BudgetExceeded("A28_ZERO_NEW_LABEL_CAP")
        if category == "full_wave_material_states":
            if limit is not None:
                _state_cap(limit)
            limit = self.full_wave_limit if limit is None else min(limit, self.full_wave_limit)
        elif limit is None:
            raise ValueError("EXPLICIT_CAP_REQUIRED_FOR_OTHER_ACTIONS")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("NONNEGATIVE_INTEGER_ACTION_CAP_REQUIRED")
        used = self.prior_reservations[category] + self.reservations[category]
        if used >= limit:
            raise BudgetExceeded("CUMULATIVE_ACTION_CAP:" + category)
        self.reservations[category] += 1
        self.append({"event": "RESERVED", "category": category,
                     "job_used": self.reservations[category], "used": used + 1,
                     "limit": limit, "status": "RESERVED", **self.metadata})

    @contextmanager
    def span(self, label, **counters):
        """Centralize inherited real full forwards before their first operation.

        New code must choose this counted span or an explicit state reservation,
        rather than both. Historical audit's manually paid two SciPy states are
        retained as recorded, and cannot be rerun after the cap violation.
        """
        calls = counters.get("full_forward_calls", 0)
        if isinstance(calls, bool) or not isinstance(calls, int) or calls < 0:
            raise ValueError("NONNEGATIVE_FULL_FORWARD_SPAN_COUNT_REQUIRED")
        for _ in range(calls):
            self.reserve("full_wave_material_states")
            self.central_full_forward_span_reservations += 1
        with super().span(label, **counters) as row:
            yield row

    def receipt(self):
        # Sample at receipt boundaries as well as periodic checks so short jobs
        # retain their final GPU/CPU memory measurements.
        resource_sample_error = None
        try:
            super().check()
        except Exception as exc:
            # Failed device initialization must still produce a durable terminal
            # receipt; retain unavailable resource diagnostics without hiding it.
            resource_sample_error = type(exc).__name__ + ": " + str(exc)
        row = {**self.metadata, **super().receipt()}
        if resource_sample_error is not None:
            row["resource_sample_error"] = resource_sample_error
        wall, process, cpu = self.elapsed()
        row.update(wall_seconds=wall, process_cpu_seconds=process,
                   charged_cpu_seconds=cpu,
                   device=self.accounting_device,
                   gpu_occupation_seconds=wall if self.accounting_device == "cuda" and not self.gpu_host_only else 0.0,
                   gpu_host_only=self.gpu_host_only,
                   raw_monitor_wall_seconds=wall if self.gpu_host_only else None,
                   durable_reservations=dict(self.reservations),
                   central_full_forward_span_reservations=self.central_full_forward_span_reservations,
                   full_wave_material_states=self.reservations["full_wave_material_states"],
                   cumulative_full_wave_material_states=(self.prior_reservations["full_wave_material_states"]+
                                                          self.reservations["full_wave_material_states"]),
                   full_wave_material_states_max=self.full_wave_limit,
                   cumulative_campaign_full_wave_material_states=(self.prior_campaign_states+
                                                                    self.reservations["full_wave_material_states"]),
                   prior_legacy_unreserved_real_fixture_states=self.prior_legacy_fixture_states,
                   physics_disabled_evidence_only=self.evidence_only,
                   prior_cpu_seconds=self.prior_cpu, prior_gpu_seconds=self.prior_gpu,
                   cpu_limit_seconds=self.cpu_limit, gpu_limit_seconds=self.gpu_limit,
                   cpu_accounting_rule="max(wall,processCPU) on CPU; processCPU on CUDA",
                   gpu_accounting_rule="entire CUDA job occupied wall")
        return row


@contextmanager
def charged_job(root, job, *, device="cpu", config=None, metadata=None,
                shared_lock_path=None, gpu_host_only=False):
    """Start a new paid attempt; retries use new IDs and inherit all costs."""
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    root, job = Path(root).resolve(), str(job)
    if not job or Path(job).name != job or job in {".", ".."} or any(c in job for c in "/\\\0"):
        raise ValueError("SIMPLE_JOB_ID_REQUIRED")
    config = dict(config or {})
    if config.get("new_labels", False) is not False or config.get("new_full_forward_labels_max", 0) != 0:
        raise ValueError("A28_LABEL_CAP_CANNOT_BE_EXPANDED")
    if any(config.get(key, False) is not False for key in
           ("train", "new_training", "training_enabled", "new_training_labels")):
        raise ValueError("A28_E1_NO_TRAINING")
    if config.get("locked_enabled", False) is not False:
        raise ValueError("A28_E1_LOCKED_DATA_CLOSED")
    state_limit = _state_cap(config.get("full_wave_material_states_max", MAX_FULL_WAVE_MATERIAL_STATES))
    if device not in {"cpu", "cuda"}:
        raise ValueError("A28_CPU_OR_CUDA_REQUIRED")
    cpu_limit = float(config.get("cpu_budget_seconds", config.get("cpu_limit_seconds", CPU_LIMIT_SECONDS)))
    gpu_limit = float(config.get("gpu_budget_seconds", config.get("gpu_limit_seconds", GPU_LIMIT_SECONDS)))
    if not 0 < cpu_limit <= CPU_LIMIT_SECONDS or not 0 < gpu_limit <= GPU_LIMIT_SECONDS:
        raise ValueError("A28_BUDGET_CANNOT_BE_EXPANDED")
    for name, reserve, limit in (("cpu_stop_reserve_seconds", config.get("cpu_stop_reserve_seconds", 15.0), cpu_limit),
                                 ("gpu_stop_reserve_seconds", config.get("gpu_stop_reserve_seconds", 30.0), gpu_limit)):
        if not math.isfinite(float(reserve)) or not 0 <= float(reserve) < limit:
            raise ValueError("INVALID_A28_STOP_RESERVE:" + name)
    owner = "a28:" + job + ":" + uuid.uuid4().hex
    with owned_file_lock(root / "runs/a28-accounting.lock", owner_id=owner, job=job) as accounting_lock:
        history = budget_history(root)
        if history["unresolved_jobs"]:
            raise BudgetExceeded("UNACCOUNTED_INTERRUPTED_JOBS:" + ",".join(history["unresolved_jobs"]))
        used_states = history["reservations"].get("full_wave_material_states", 0)
        if used_states > state_limit and not _evidence_only(device, metadata):
            raise BudgetExceeded("A28_FROZEN_STATE_CAP_BELOW_EXISTING_USAGE")
        directory = root / "results/a28/jobs" / job
        if directory.exists():
            raise BudgetExceeded("JOB_ID_ALREADY_EXISTS; terminal receipts are immutable")
        directory.mkdir(parents=True)
        manifest = {**(metadata or {}), "job": job, "device": device, "owner_id": owner,
                    "status": "STARTED", "started_utc":
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "prior_cpu_seconds": history["cpu_seconds"],
                    "prior_gpu_seconds": history["gpu_seconds"],
                    "prior_reservations": history["reservations"]}
        write_json(directory / "job_manifest.json", manifest)
        book = BudgetLedger(directory / "COST_LEDGER.jsonl", device=device,
                            prior_cpu=history["cpu_seconds"], prior_gpu=history["gpu_seconds"],
                            prior_reservations=history["reservations"],
                            cpu_limit=cpu_limit, gpu_limit=gpu_limit,
                            cpu_reserve=config.get("cpu_stop_reserve_seconds", 15.0),
                            gpu_reserve=config.get("gpu_stop_reserve_seconds", 30.0),
                            metadata={**(metadata or {}), "job": job, "owner_id": owner},
                            checkpoint_path=directory / "job_checkpoint.json",
                            started_wall=started_wall, started_cpu=started_cpu,
                            gpu_host_only=gpu_host_only,
                            prior_campaign_states=history["campaign_full_wave_material_states"],
                            prior_legacy_fixture_states=history["legacy_unreserved_real_fixture_states"])
        book._owned_locks = [accounting_lock]
        book.full_wave_limit = state_limit
        status, error = "COMPLETE", None
        try:
            if device == "cuda" and not gpu_host_only:
                path = SHARED_GPU_LOCK if shared_lock_path is None else Path(shared_lock_path)
                if os.name == "nt" and path.as_posix().lower() != SHARED_GPU_LOCK.as_posix().lower():
                    raise BudgetExceeded("A28_REQUIRES_A20_SHARED_GPU_LOCK")
                with owned_file_lock(path, owner_id=owner, job=job) as gpu_lock:
                    book._owned_locks.append(gpu_lock)
                    book.check()
                    yield book
                    book.check()
                    book._owned_locks.pop()
            else:
                book.check()
                yield book
                book.check()
        except BaseException as exc:
            status = ("BUDGET_STOP" if isinstance(exc, BudgetExceeded) else
                      "INTERRUPTED" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "FAILED")
            error = type(exc).__name__ + ": " + str(exc)
            raise
        finally:
            receipt = book.receipt()
            receipt.update(status=status, error=error, receipt_version=2,
                           ended_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            path = directory / "job_receipt.json"
            # Never replace a terminal receipt even if another caller interferes.
            if path.exists():
                raise BudgetExceeded("TERMINAL_RECEIPT_ALREADY_EXISTS")
            write_json(path, receipt)


Ledger = BudgetLedger
job_lock = owned_file_lock
