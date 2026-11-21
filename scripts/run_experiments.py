"""Run all Phase II experiments and write CSV tables, figures and report.md (default: results/synthetic_weather/).

Usage:
    python scripts/run_experiments.py           # full run
    python scripts/run_experiments.py --quick   # fast check (fewer days and seeds)
    python scripts/run_experiments.py --weather data/real/reunion_weather_1min.csv --out results/real_weather
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
    ap.add_argument("--weather", default=None, help="measured weather CSV, e.g. data/real/reunion_weather_1min.csv")
    a = ap.parse_args()
    run_all(quick=a.quick, out_dir=a.out, weather=a.weather)
