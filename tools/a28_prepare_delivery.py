"""Prepare a truthful A28 E1 public archive from saved evidence only.

No solver, labels, training, locked inputs, connection secrets or hash checks.
An optional exposed-input bundle must be requested explicitly. Saved exact-state
LU caches are runtime assets and are excluded from every public archive.
"""
from __future__ import annotations

import time
START_WALL, START_CPU = time.perf_counter(), time.process_time()
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from a28.budget import (budget_history, charged_job, owned_file_lock, write_json,
                        MAX_FULL_WAVE_MATERIAL_STATES)

FROZEN_A27 = "325cc8a5f99d686ce1b1570296f94b0df1e90141"
REPORTS = ("README.md", "E1_DECISION.md")
PUBLICATION = ROOT / "publication"
FORBIDDEN_PARTS = frozenset({"private", "locked", "locked_id", "locked_ood", "exact_states"})
SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----"),
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(rb"AKIA[A-Z0-9]{16}"),
    re.compile(rb"(?<![A-Za-z0-9])100\.(?:[0-9]{1,3}\.){2}[0-9]{1,3}(?![0-9])"),
)


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def register_external_audits(root=ROOT):
    """Import missing declared receipts once; preserve every original receipt."""
    root = Path(root)
    with owned_file_lock(root / "runs/a28-accounting.lock",
                         owner_id="a28-delivery-import:" + uuid.uuid4().hex,
                         job="a28-delivery-audit-import"):
        edits = root / "research/delegated/a28_edits/TEST_RECEIPT.json"
        directory = root / "results/a28/jobs/a28-worker-edits-tests-001"
        if edits.exists() and not (directory / "job_receipt.json").exists():
            if directory.exists():
                raise ValueError("EXTERNAL_RECEIPT_DIRECTORY_ALREADY_EXISTS")
            row = _json(edits)
            if row.get("physics_actions") != 0 or row.get("locked_reads") != 0 or row.get("new_labels") != 0:
                raise ValueError("SYNTHETIC_WORKER_IMPORT_MUST_HAVE_ZERO_PHYSICS")
            wall = float(row["process_wall_seconds"])
            process = float(row["process_user_seconds"]) + float(row["process_system_seconds"])
            write_json(directory / "job_manifest.json", {"job": directory.name,
                "stage": "accounting", "status": "EXTERNAL_MEASURED_IMPORT",
                "source": edits.relative_to(root).as_posix()})
            write_json(directory / "job_receipt.json", {"job": directory.name, "stage": "accounting",
                "status": "COMPLETE", "device": "cpu", "wall_seconds": wall,
                "process_cpu_seconds": process, "charged_cpu_seconds": max(wall, process),
                "gpu_occupation_seconds": 0., "counts": {"synthetic_unit_tests": row["tests"]},
                "durable_reservations": {}, "source": edits.relative_to(root).as_posix(),
                "external_measured_receipt": True, "physics_started": False})
        directory = root / "results/a28/jobs/a28-headroom-001-startup-upper-bound"
        if not (directory / "job_receipt.json").exists():
            if directory.exists():
                raise ValueError("STARTUP_AUDIT_DIRECTORY_ALREADY_EXISTS")
            write_json(directory / "job_manifest.json", {"job": directory.name,
                "stage": "accounting", "status": "FAILED_BEFORE_NATIVE_CONTEXT",
                "measurement": "declared conservative one-second CPU upper bound"})
            write_json(directory / "job_receipt.json", {"job": directory.name, "stage": "accounting",
                "status": "FAILED", "device": "cpu", "wall_seconds": 1.,
                "process_cpu_seconds": 0., "charged_cpu_seconds": 1., "gpu_occupation_seconds": 0.,
                "counts": {}, "durable_reservations": {}, "physics_started": False,
                "accounting_kind": "CONSERVATIVE_UPPER_BOUND_NOT_NATIVE_TIMING",
                "error": "a28-headroom-001 blocked at startup before native context; no physics; parent-authorized one-second conservative audit",
                "original_native_receipt": "ABSENT; never manufactured or overwritten"})


def _write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def render_accounting(cutoff, active_job=None, *, root=ROOT):
    """Expose terminal costs and the known all-resolution budget violation."""
    root = Path(root)
    history = budget_history(root)
    costs = [{"event": "TERMINAL_JOB", "accounting_cutoff": cutoff, **row}
             for row in history["receipts"]]
    failures = []
    for row in history["receipts"]:
        if row["status"] != "COMPLETE":
            failures.append({"kind": "JOB_FAILURE", "job": row["job"], "status": row["status"],
                "error": row.get("error"), "accounting_kind": row.get("accounting_kind", "NATIVE_RECEIPT"),
                "charged_cpu_seconds": row.get("charged_cpu_seconds"),
                "gpu_seconds": row["gpu_occupation_seconds"],
                "source": "results/a28/jobs/" + row["job"] + "/job_receipt.json"})
    for folder in ("anatomy", "headroom", "posterior", "bp"):
        path = root / "results/a28" / folder / "FAILURES.json"
        if path.exists():
            for row in _json(path):
                failures.append({"kind": "RETAINED_CASE_FAILURE", "attempt_folder": folder, **row})
    for path in sorted((root / "results/a28/transport").glob("*.json")):
        row = _json(path)
        if str(row.get("status", "")).startswith(("FAILED", "ERROR", "BUDGET_STOP")):
            item = {"kind": "RETURNED_REMOTE_FAILURE", "source": path.relative_to(root).as_posix(),
                    "status": row["status"], "retained_returned_result": row,
                    "native_receipt_not_rewritten": True}
            if path.name == "a28-transport-inventory-999ab9d04fc9.json":
                item["receipt_status_mismatch"] = "original transport receipt COMPLETE; returned bootstrap result FAILED"
                item["cause"] = "predictions archive needed explicit validation role; fixed before retry"
            if path.name == "a28-transport-push-6c17f853ad12.json":
                item["cause"] = "Windows CRLF serialization mismatch on frozen transfer; failed attempt retained"
            failures.append(item)
    strict = history["strict_all_resolution_full_wave_material_states"]
    cap = MAX_FULL_WAVE_MATERIAL_STATES
    if history["legacy_unreserved_real_fixture_states"] or strict > cap:
        failures.append({"kind": "FULL_WAVE_ACCOUNTING_COVERAGE_DEFECT",
            "status": history["full_wave_material_state_budget_compliance"],
            "campaign_centrally_reserved_states": history["campaign_full_wave_material_states"],
            "legacy_unreserved_real_tiny_fixture_states": history["legacy_unreserved_real_fixture_states"],
            "strict_all_resolution_known_states": strict, "strict_count_label": history["strict_state_count_label"],
            "state_cap": cap, "known_excess_states": max(0, strict - cap),
            "fixture_jobs": history["legacy_real_fixture_jobs"],
            "unclassified_legacy_dense_oracle_audits": history["unclassified_legacy_dense_oracle_audits"],
            "source_audit": "research/delegated/a28_infrastructure/SOURCE_REVIEW.md",
            "policy": "new physics and GPU experiments stopped; only CPU saved-evidence delivery permitted",
            "historical_receipts_and_frozen_config_unchanged": True})
    for name, rows in (("A28_COST_LEDGER.jsonl", costs), ("A28_FAILURE_LEDGER.jsonl", failures)):
        _write_jsonl(root / name, rows)
    _write_jsonl(root / "results/a28/COST_LEDGER.jsonl", costs)
    _write_jsonl(root / "results/a28/FAILURE_LEDGER.jsonl", failures)
    config = _json(root / "a28_config.json")
    unresolved = [job for job in history["unresolved_jobs"] if job != active_job]
    summary = {"cutoff": cutoff, "cpu_charged_seconds": history["cpu_seconds"],
        "gpu_occupied_seconds": history["gpu_seconds"], "cpu_cap_seconds": config["cpu_limit_seconds"],
        "gpu_cap_seconds": config["gpu_limit_seconds"], "accepted_terminal_receipts": len(history["receipts"]),
        "unresolved_jobs": unresolved, "active_job_excluded_at_cutoff": active_job,
        "failure_records": len(failures), "new_training_labels": 0, "training": "NOT_RUN", "locked_ID_OOD": "UNOPENED",
        "campaign_full_wave_material_states": history["campaign_full_wave_material_states"],
        "legacy_unreserved_real_fixture_states": history["legacy_unreserved_real_fixture_states"],
        "strict_all_resolution_full_wave_material_states": strict,
        "strict_state_count_label": history["strict_state_count_label"],
        "strict_state_total_is_known_lower_bound": history["strict_state_total_is_known_lower_bound"],
        "unclassified_legacy_dense_oracle_audits": history["unclassified_legacy_dense_oracle_audits"],
        "full_wave_material_states_cap": cap, "known_state_excess": max(0, strict - cap),
        "full_wave_budget_compliance": history["full_wave_material_state_budget_compliance"],
        "cpu_budget_compliance": "PASS" if history["cpu_seconds"] <= config["cpu_limit_seconds"] else "FAIL",
        "gpu_budget_compliance": "PASS" if history["gpu_seconds"] <= config["gpu_limit_seconds"] else "FAIL",
        "protocol_acceptance": "BUDGET_FAIL" if strict > cap else "NOT_A_SCIENTIFIC_GATE_VERDICT",
        "cpu_rule": "CPU max(wall,process CPU); CUDA host process CPU; declared startup upper bound explicit",
        "gpu_rule": "entire CUDA occupied wall, including setup, diagnostics and failures",
        "action_count_rule": "material states separate from all RHS/JVP/adjoint/operator actions; no aggregate double count",
        "publication_is_independent_of_scientific_GO": True}
    write_json(root / "results/a28/COST_SUMMARY.json", summary)
    return summary


def public_file(path, *, root=ROOT, allow_npz=False):
    """Fail closed on public paths and secret contents; never print matches."""
    root, path = Path(root).resolve(), Path(path)
    relative = path.relative_to(root)
    if path.is_symlink() or any(part.lower() in FORBIDDEN_PARTS for part in relative.parts):
        raise ValueError("FORBIDDEN_PUBLIC_PATH:" + relative.as_posix())
    if path.suffix in {".npz", ".pt", ".npy"} and not allow_npz:
        raise ValueError("BINARY_ARRAYS_MUST_USE_RELEASE:" + relative.as_posix())
    if path.suffix not in {".npz", ".npy", ".pt", ".png", ".jpg", ".jpeg", ".zip"}:
        if any(pattern.search(path.read_bytes()) for pattern in SECRET_PATTERNS):
            raise ValueError("PUBLIC_SECRET_SCAN_REJECTED_FILENAME:" + relative.as_posix())
    return relative.as_posix()


def raw_bundle(book, *, active_job=None, root=ROOT):
    root = Path(root)
    destination = root / "publication/A28_E1_RAW_EVIDENCE.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    members = []
    for path in sorted((root / "results/a28").rglob("*")):
        if not path.is_file() or "exact_states" in path.parts or ".tmp-" in path.name or path.suffix == ".tmp":
            continue
        if active_job and active_job in path.parts:
            continue
        if path.name == "job_checkpoint.json":
            continue
        public_file(path, root=root, allow_npz=True)
        members.append(path)
    with book.span("a28_saved_evidence_archive", archives=1), zipfile.ZipFile(
            destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for i, path in enumerate(members):
            if i % 50 == 0:
                book.check()
            archive.write(path, path.relative_to(root).as_posix())
    return destination, len(members)


def _lawful_input(root, name):
    root = Path(root).resolve()
    path = root / name
    if (path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root)
            or any("locked" in part.lower() or part.lower() == "private" for part in path.parts)):
        raise ValueError("EXPOSED_INPUT_REQUIRED_NO_LOCKED_OR_PRIVATE:" + name)
    return path


def exposed_input_bundle(book, data_root, a26_root):
    """Explicit, validation-only dependencies; TRAIN/locked payloads never open."""
    import numpy as np
    from a26.data import load_split
    data_root, a26_root = Path(data_root).resolve(), Path(a26_root).resolve()
    with book.span("a28_explicit_exposed_input_validation", exposed_validation_imports=1):
        data = load_split(data_root, "validation")
        expected = np.arange(250065, 250081)
        if not np.array_equal(np.unique(data["parent_id"]), expected):
            raise ValueError("EXPOSED_VALIDATION_PARENT_WHITELIST_REQUIRED")
        manifest = _json(_lawful_input(data_root, "manifest.json"))
        manifest["parents"] = [row for row in manifest["parents"] if row.get("split") == "validation"]
        manifest["splits"] = {"validation": list(map(int, expected))}
        manifest["public_payload_scope"] = "exposed development only; TRAIN and locked payloads excluded"
        dependencies = [("inputs/a25/data/validation.npz", _lawful_input(data_root, "validation.npz")),
            ("inputs/a25/data/geometry.json", _lawful_input(data_root, "geometry.json"))]
        for bg in ("weak", "ring"):
            path = _lawful_input(data_root, "background/" + bg + ".npz")
            with np.load(path, allow_pickle=False) as archive:
                if any("truth" in key.lower() or "locked" in key.lower() for key in archive.files):
                    raise ValueError("KNOWN_BACKGROUND_CACHE_REQUIRED")
                descriptor = json.loads(str(archive["closure_provenance"].item()))
                if descriptor.get("background_id") != bg:
                    raise ValueError("KNOWN_BACKGROUND_PROVENANCE_MISMATCH")
            dependencies.append(("inputs/a25/data/background/" + bg + ".npz", path))
        source_config = _lawful_input(data_root.parent, "configs/a25/nn_admitted.json")
        dependencies.append(("inputs/a25/configs/a25/nn_admitted.json", source_config))
        names = ("results/a26/cache/validation_N3.npz",
                 "results/a26/cache/validation_ONLINE_MEASUREMENTS.npz",
                 "results/a26/evaluation/261009/PREDICTIONS.npz",
                 "results/a26/learning/261009/N3/best.pt")
        for name in names:
            path = _lawful_input(a26_root, name)
            if path.suffix == ".npz":
                with np.load(path, allow_pickle=False) as archive:
                    if "case_id" not in archive.files or not np.array_equal(archive["case_id"], data["case_id"]):
                        raise ValueError("A26_EXPOSED_INPUT_EXACT_CASE_ALIGNMENT_REQUIRED")
                    if name.endswith("validation_ONLINE_MEASUREMENTS.npz"):
                        # The frozen online-only archive intentionally omits
                        # parent IDs and labels. Bind its exact case sequence
                        # to the already verified validation bundle instead.
                        if set(archive.files) != {"data", "sigma", "background_id", "case_id"}:
                            raise ValueError("A26_MEASUREMENT_ONLY_ARCHIVE_REQUIRED")
                        for key in ("sigma", "background_id"):
                            if not np.array_equal(archive[key], data[key]):
                                raise ValueError("A26_ONLINE_ARCHIVE_METADATA_MISMATCH")
                    elif ("parent_id" not in archive.files or
                          not np.array_equal(archive["parent_id"], data["parent_id"]) or
                          not np.array_equal(np.unique(archive["parent_id"]), expected)):
                        raise ValueError("A26_EXPOSED_INPUT_PARENT_WHITELIST_REQUIRED")
                    if "split" in archive.files and set(map(str, archive["split"])) != {"validation"}:
                        raise ValueError("A26_EXPOSED_INPUT_SPLIT_MISMATCH")
            dependencies.append(("inputs/a26/" + name, path))
        destination = PUBLICATION / "A28_E1_EXPOSED_INPUTS.zip"
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("inputs/a25/data/manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
            for member, path in dependencies:
                book.check()
                archive.write(path, member)
            archive.writestr("INPUT_SCOPE.json", json.dumps({"status": "EXPOSED_DEVELOPMENT_ONLY",
                "parents": list(map(int, expected)), "new_labels": 0, "TRAIN_payload": "EXCLUDED",
                "locked_ID_OOD_payload": "EXCLUDED", "checkpoint": "frozen A26 N3 261009",
                "A25_commit": "59d8cc57c9da99ccb7978febc308193c43d6f3e5",
                "A26_commit": "2603f3bd16c48e48587fedc4363975779def17f5"}, indent=2) + "\n")
    return destination


def check_frozen_delivery(book, *, root=ROOT):
    root = Path(root)
    config = _json(root / "a28_config.json")
    if config.get("source_commit") != FROZEN_A27 or _json(root / "results/a28/FROZEN_PROTOCOL.json") != config:
        raise ValueError("FROZEN_A28_PROTOCOL_MISMATCH")
    book.check()
    check = subprocess.run(["git", "diff", "--quiet", FROZEN_A27, "--", "src/a24", "src/a25",
                            "src/a26", "src/a27", "vendor"], cwd=root, capture_output=True, timeout=20)
    if check.returncode:
        raise ValueError("FROZEN_BACKEND_OR_LEARNING_DEPENDENCY_CHANGED")
    book.check()
    artifacts = _json(root / "results/a28/artifacts/SUMMARY.json")
    if artifacts.get("status") not in {"ARTIFACTS_WRITTEN", "COMPLETE"}:
        raise ValueError("SAVED_EVIDENCE_ARTIFACTS_REQUIRED")
    broken = []
    for name in REPORTS:
        path = root / name
        public_file(path, root=root)
        for target in re.findall(r"\]\(([^\s)]+)\)", path.read_text(encoding="utf-8")):
            if target.startswith(("https://", "http://", "#")):
                continue
            if not (root / target.split("#")[0]).exists():
                broken.append({"file": name, "link": target})
    if broken:
        raise ValueError("BROKEN_LOCAL_REPORT_LINKS:" + json.dumps(broken))
    return config, artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job")
    parser.add_argument("--include-exposed-inputs", action="store_true")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--a26-root", type=Path)
    args = parser.parse_args()
    if args.include_exposed_inputs and (args.data_root is None or args.a26_root is None):
        parser.error("explicit local --data-root and --a26-root required; no automatic remote access")
    register_external_audits()
    config = _json(ROOT / "a28_config.json")
    with charged_job(ROOT, args.job, config=config, metadata={"stage": "delivery-preparation"}) as book:
        book.deadline -= book.started_wall - START_WALL
        book.started_wall, book.started_cpu = START_WALL, START_CPU
        cost = render_accounting("science-complete-before-delivery", active_job=args.job)
        config, artifacts = check_frozen_delivery(book)
        if args.include_exposed_inputs:
            inputs = exposed_input_bundle(book, args.data_root, args.a26_root)
        else:
            inputs = None
        raw, members = raw_bundle(book, active_job=args.job)
        write_json(ROOT / "A28_SOURCE_MANIFEST.json", {"frozen_A27_commit": FROZEN_A27,
            "frozen_A26_commit": config["a26_commit"], "frozen_A25_commit": config["a25_commit"],
            "inherited_backend_sources_unchanged": True, "new_source": "src/a28",
            "physics": "24x24 2D TM; full-grid complex material; eight sources, thirty-two receivers",
            "raw_evidence_asset": raw.name, "raw_bytes": raw.stat().st_size, "archive_members": members,
            "exposed_input_asset": inputs.name if inputs else "NOT_BUNDLED; reuse explicit A25/A26 dependencies",
            "exact_state_LU_caches": "EXCLUDED_RUNTIME_CACHE", "training_payload": "EXCLUDED", "locked_payload": "EXCLUDED",
            "new_training": "NOT_RUN", "new_labels": 0, "SHA256_checks": "NOT_RUN_BY_USER_POLICY",
            "reports_checked": list(REPORTS), "artifact_status": artifacts["status"],
            "scientific_status": "Read E1_DECISION.md; publication does not imply GO",
            "strict_full_wave_known_state_count": cost["strict_all_resolution_full_wave_material_states"],
            "strict_count_label": cost["strict_state_count_label"], "budget_compliance": cost["full_wave_budget_compliance"],
            "replication_commit": "release target actual git commit"})
        print(json.dumps({"status": "PREPARED", "raw_asset": raw.name, "bytes": raw.stat().st_size,
                          "members": members, "budget_compliance": cost["full_wave_budget_compliance"]}), flush=True)
    # The finished preparation receipt now exists. Charge this saved-evidence
    # accounting tail too; its own terminal receipt follows the stated cutoff.
    tail_job = args.job + "-accounting"
    with charged_job(ROOT, tail_job, config=config, metadata={"stage": "accounting"}):
        render_accounting("preparation-complete-before-publication", active_job=tail_job)


if __name__ == "__main__":
    main()
