"""Telemetry twin: real-or-synthetic weather -> healthy device telemetry -> injected faults -> labels."""

from __future__ import annotations

import pandas as pd

from ..labels import make_labels
from ..profiles import Profile
from .base import make_weather
from .drive import healthy_drive
from .faults import inject_faults
from .inverter import healthy_inverter


def generate(
    profile: Profile | str, days: int, seed: int, n_true: int = 8, n_near: int = 4, weather: pd.DataFrame | None = None
) -> dict:
    """Return raw telemetry, pu telemetry, labels and the injected episode list for one asset."""
    if isinstance(profile, str):
        profile = Profile.load(profile)
    w = weather if weather is not None else make_weather(days, seed)
    healthy = (
        healthy_inverter(profile, w, seed)
        if profile.asset_type == "hybrid_inverter"
        else healthy_drive(profile, w, seed)
    )
    raw, episodes = inject_faults(profile, healthy, seed, n_true=n_true, n_near=n_near)
    pu = profile.to_pu(raw)
    labels = make_labels(profile, pu)
    return {"profile": profile, "raw": raw, "pu": pu, "labels": labels, "episodes": episodes}
