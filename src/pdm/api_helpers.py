"""Small helper to build a Platform for any profile, training the model on first use."""

from __future__ import annotations

import pickle
from pathlib import Path

from .agents.orchestrator import Platform
from .profiles import Profile
from .training import ROOT, train_and_save


def platform_for(profile_name: str, model_dir: str | Path | None = None) -> Platform:
    profile = Profile.load(profile_name)
    base = "inverter_hybrid_5kw" if profile.asset_type == "hybrid_inverter" else "pump_drive_5hp"
    path = Path(model_dir or ROOT / "results" / "models") / f"{base}_ours.pkl"
    if not path.exists():
        path = train_and_save(base, out_dir=path.parent)
    with open(path, "rb") as f:
        return Platform(profile, pickle.load(f))
