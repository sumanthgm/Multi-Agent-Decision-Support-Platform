"""Run all Phase II experiments and write results/ (CSV tables, figures, report.md).

Usage:
    python scripts/run_experiments.py           # full run
    python scripts/run_experiments.py --quick   # fast check (fewer days and seeds)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pdm.experiments import run_all  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run_all(quick=a.quick, out_dir=a.out)
