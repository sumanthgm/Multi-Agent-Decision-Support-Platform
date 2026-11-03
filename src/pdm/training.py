"""Train and save the Early-Warning model for one asset type (used by the API and scripts/train.py)."""

from __future__ import annotations

import pickle
from pathlib import Path

from .agents.early_warning import EarlyWarningXGB
from .features import pu_features, targets, window_ends
from .twin import generate

ROOT = Path(__file__).resolve().parents[2]


def train_and_save(
    profile_name: str,
    days: int = 60,
    seed: int = 1,
    n_true: int = 10,
    n_near: int = 5,
    out_dir: str | Path | None = None,
) -> Path:
    d = generate(profile_name, days, seed, n_true=n_true, n_near=n_near)
    P = d["profile"]
    e = window_ends(d["pu"].index)
    y, m = targets(P, d["labels"], e)
    model = EarlyWarningXGB(P.alarm_codes).fit(pu_features(P, d["pu"], e), y, m)
    out = Path(out_dir) if out_dir else ROOT / "results" / "models"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{profile_name}_ours.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path
