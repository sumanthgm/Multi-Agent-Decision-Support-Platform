"""Telemetry twin: real-or-synthetic weather -> healthy device telemetry -> injected faults -> labels."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from ..labels import make_labels
from ..profiles import Profile
from .base import load_weather, make_weather, weather_block
from .drive import healthy_drive
from .faults import inject_faults
from .inverter import healthy_inverter


@lru_cache(maxsize=2)
def _measured(path: str) -> pd.DataFrame:
    return load_weather(path)


def generate(
    profile: Profile | str,
    days: int,
    seed: int,
    n_true: int = 8,
    n_near: int = 4,
    weather: pd.DataFrame | str | Path | None = None,
    weather_day: int = 0,
) -> dict:
    """Return raw telemetry, pu telemetry, labels and the injected episode list for one asset.

    weather: None for synthetic weather, a weather DataFrame, or the path of a measured 1-minute
    weather CSV, from which `days` complete days starting at day number `weather_day` are used.
    """
    if isinstance(profile, str):
        profile = Profile.load(profile)
    if weather is None:
        w = make_weather(days, seed)
    elif isinstance(weather, (str, Path)):
        w = weather_block(_measured(str(weather)), weather_day, days)
    else:
        w = weather
    healthy = (
        healthy_inverter(profile, w, seed)
        if profile.asset_type == "hybrid_inverter"
        else healthy_drive(profile, w, seed)
    )
    raw, episodes = inject_faults(profile, healthy, seed, n_true=n_true, n_near=n_near)
    pu = profile.to_pu(raw)
    labels = make_labels(profile, pu)
    return {"profile": profile, "raw": raw, "pu": pu, "labels": labels, "episodes": episodes}
