"""CPU-only A28 arithmetic and guard verification after experiment closure.

No inherited real-backend fixture discovery, data imports or Maxwell execution.
The evidence-only ledger blocks physical reservations before an operation.
"""
from __future__ import annotations

import time
START_WALL, START_CPU = time.perf_counter(), time.process_time()
import argparse
import ast
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from a28.budget import charged_job, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", help="new immutable CPU evidence receipt ID")
    args = parser.parse_args()
    config = json.loads((ROOT / "a28_config.json").read_text(encoding="utf-8"))
    with charged_job(ROOT, args.job, config=config, metadata={"stage": "evidence-report"}) as book:
        book.deadline -= book.started_wall - START_WALL
        book.started_wall, book.started_cpu = START_WALL, START_CPU
        source_files = [*(ROOT / "src/a28").glob("*.py"),
                        *(ROOT / "tools").glob("a28*.py"),
                        *(ROOT / "tests").glob("test_a28*.py")]
        for path in source_files:
            ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
        from a28_driver import tests_stage
        result = tests_stage(ROOT, book, pattern="test_a28*.py")
        if book.reservations.get("full_wave_material_states", 0) or book.counts.get("full_forward_calls", 0):
            raise RuntimeError("NO_NEW_REAL_PHYSICS_ALLOWED_IN_SOURCE_CHECK")
        result.update(real_backend_executions=0, source_AST_files=len(source_files))
        write_json(ROOT / "results/a28/jobs" / args.job / "RESULT.json", result)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
