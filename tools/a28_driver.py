"""Foreground paid A28 mechanism stages; fixed development inputs and caps."""
from __future__ import annotations

import time
START_WALL, START_CPU = time.perf_counter(), time.process_time()

import argparse
import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("audit", "micro", "freeze", "anatomy", "headroom", "posterior", "bp", "report", "tests", "theory")
sys.path.insert(0, str(ROOT / "src"))


def default_data_root():
    return (Path("D:/AI/A25_ADAPTIVE_OM_FEEDBACK/data") if os.name == "nt" else
            ROOT.parents[1] / "A25/a25-adaptive-om-feedback/data")


def default_a26_root():
    return (Path("D:/AI/A26_OM_FEEDBACK_PRIOR_COMPLETION") if os.name == "nt" else
            ROOT.parents[1] / "A26/a26-om-feedback-prior-completion")


def tests_stage(root, book, pattern="test_a28*.py"):
    import unittest
    from a24.contracts import active_ledger
    from a28.budget import write_json

    directory = Path(root) / "results/a28/jobs" / book.metadata["job"]
    directory.mkdir(parents=True, exist_ok=True)
    with active_ledger(book), book.span("source_unit_tests"):
        suite = unittest.defaultTestLoader.discover(str(Path(root) / "tests"), pattern=pattern)
        with (directory / "TESTS.log").open("w", encoding="utf-8") as log:
            result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
        output = {"tests": result.testsRun, "errors": len(result.errors),
                  "failures": len(result.failures), "skipped": len(result.skipped),
                  "skipped_tests": [{"test": str(test), "reason": str(reason)}
                                    for test, reason in result.skipped],
                  "passed": result.wasSuccessful(),
                  "scope": "unittest filename pattern=" + pattern}
        write_json(directory / "TESTS_SUMMARY.json", output)
        if not result.wasSuccessful():
            raise RuntimeError("UNIT_TESTS_FAILED:" + json.dumps(output))
    return output


def theory_stage(root, book):
    """Execute the supplied algebra source from a new paid job directory.

    The specification package's pre-existing JSON is never overwritten.
    Assertions are taken from the supplied source without changing its math.
    """
    import runpy
    from contextlib import redirect_stdout
    from a28.budget import write_json
    root = Path(root)
    candidates = (root / "theory/a28/A28_theory_checks.py",
                  root / "theory/A28_theory_checks.py",
                  root.parent / "A28_research_package/A28_theory_checks.py")
    source = next((p for p in candidates if p.is_file()), None)
    if source is None:
        raise RuntimeError("SUPPLIED_A28_THEORY_SOURCE_UNAVAILABLE")
    directory = root / "results/a28/jobs" / book.metadata["job"]
    copy = directory / "A28_theory_checks.py"
    with book.span("supplied_a28_algebra_checks"):
        copy.write_bytes(source.read_bytes())
        with (directory / "THEORY.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
            runpy.run_path(str(copy), run_name="__main__")
        result = json.loads(copy.with_suffix(".json").read_text(encoding="utf-8"))
        checks = {k: v for k, v in result.items() if isinstance(v, dict) and "passed" in v}
        if not checks or not all(v["passed"] is True for v in checks.values()):
            raise RuntimeError("SUPPLIED_A28_THEORY_CHECKS_FAILED")
        summary = {"status": "PASS", "checks": len(checks), "passed": True,
                   "scope": "supplied toy algebra assertions; no Maxwell imaging evidence",
                   "full_wave_material_states": 0, "source_copy_retained": True}
        write_json(directory / "THEORY_SUMMARY.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--data-root", default=str(default_data_root()))
    parser.add_argument("--a26-root", default=str(default_a26_root()))
    parser.add_argument("--config", default="a28_config.json")
    parser.add_argument("--job", required=True)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--test-pattern", default="test_a28*.py",
                        help="Default arithmetic/guard A28 tests; inherited physical fixtures consume the state budget.")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    data_root, a26_root = Path(args.data_root), Path(args.a26_root)
    sys.path.insert(0, str(root / "src"))
    config = json.loads((root / args.config).read_text(encoding="utf-8"))
    if args.device == "cuda":
        # This must precede CUDA resource sampling in charged_job: never reset
        # an existing CUDA primary context to change the host wait policy.
        from a25.cuda_runtime import configure_host_wait
        configure_host_wait("blocking_sync")
    from a28.budget import charged_job, write_json
    with charged_job(root, args.job, device=args.device, config=config,
                     metadata={"stage": args.stage}) as book:
        # Include import, configuration and host-wait setup in the native job.
        book.deadline -= book.started_wall - START_WALL
        book.started_wall, book.started_cpu = START_WALL, START_CPU
        book.check()
        directory = root / "results/a28/jobs" / args.job
        try:
            if args.stage == "tests":
                result = tests_stage(root, book, pattern=args.test_pattern)
            elif args.stage == "theory":
                result = theory_stage(root, book)
            else:
                from a28 import pipeline
                stage = getattr(pipeline, "stage_" + args.stage)
                result = stage(root, data_root, a26_root, config, book, args.device)
            write_json(directory / "RESULT.json", result)
            print(json.dumps({"job": args.job, "stage": args.stage, "result": result},
                             default=str, allow_nan=False), flush=True)
        except BaseException:
            (directory / "ERROR.log").write_text(traceback.format_exc(), encoding="utf-8")
            raise


if __name__ == "__main__":
    main()
