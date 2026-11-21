"""ASPIRE-style labels per alarm: Normal / Warning (within H minutes before onset) / Alarm (limit crossed)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .profiles import Profile

DEBOUNCE_MIN = 2  # limit must hold this long to count as an alarm
CLEAR_MIN = 30  # minutes clear before a new onset counts as a new event


def make_labels(profile: Profile, pu: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for code, a in profile.alarms.items():
        raw_active = (a.crossed(pu[a.signal]) & profile.condition_mask(pu, a.condition)).to_numpy()
        active = (
            pd.Series(raw_active)
            .rolling(DEBOUNCE_MIN, min_periods=DEBOUNCE_MIN)
            .min()
            .fillna(0)
            .to_numpy()
            .astype(bool)
        )
        onset = np.zeros(len(active), dtype=bool)
        last_active = -(10**9)
        for t in np.flatnonzero(active):
            if t - last_active > CLEAR_MIN:
                onset[t] = True
            last_active = t
        # warning[t] = an onset happens in (t, t+H] and the alarm is not active at t
        H = a.warn_minutes
        future = np.zeros(len(active), dtype=bool)
        for t in np.flatnonzero(onset):
            future[max(0, t - H) : t] = True
        out[f"{code}_active"] = active
        out[f"{code}_onset"] = onset
        out[f"{code}_warn"] = future & ~active
    lab = pd.DataFrame(out, index=pu.index)
    act = lab[[c for c in lab if c.endswith("_active")]].any(axis=1)
    warn = lab[[c for c in lab if c.endswith("_warn")]].any(axis=1)
    lab["state"] = np.where(act, 2, np.where(warn, 1, 0))  # 0 Normal, 1 Warning, 2 Alarm
    return lab
