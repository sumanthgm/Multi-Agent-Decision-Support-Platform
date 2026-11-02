"""Train the Early-Warning model for an asset type and save it to results/models/.

    python scripts/train.py --profile inverter_hybrid_5kw
    python scripts/train.py --profile pump_drive_5hp --days 60
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pdm.training import train_and_save  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="inverter_hybrid_5kw")
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    print("saved", train_and_save(a.profile, a.days, a.seed))
