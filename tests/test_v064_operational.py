"""Smoke tests for the consolidated v064 operational ARC system."""

import os
import sys

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from arc_loader import load_task
from v064_ubp_glm_operational import benchmark, solve_task


def main() -> int:
    data_dir = os.path.join(_PKG_ROOT, "data", "training")

    # Specific regression: marker-driven cross translation
    task = load_task(os.path.join(data_dir, "e48d4e1a.json"), name="e48d4e1a")
    result = solve_task(task)
    assert result is not None, "e48d4e1a should be solved"
    pred, solver = result
    assert solver == "cross_shift_by_markers", f"unexpected solver: {solver}"
    assert pred == task.test[0].expected_output, "prediction mismatch for e48d4e1a"

    # Whole-batch smoke benchmark
    summary = benchmark(data_dir)
    assert summary["solved"] >= 9, f"expected at least 9 solves, got {summary['solved']}"
    print(f"v064 smoke test passed: {summary['solved']}/{summary['total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
