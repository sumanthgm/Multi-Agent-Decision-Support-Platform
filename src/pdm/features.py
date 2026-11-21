"""Window features.

- `pu_features`: per-unit statistics + distance-to-limit + time-to-limit for every alarm.
  This is the rating- and asset-agnostic representation (our method).
- `raw_features`: the same statistics on raw signals, no limits (ablation baseline).
- `sequences`: raw or pu 1-minute sequences for the ASPIRE-style Transformer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .profiles import Profile

WINDOW = 60   # minutes of history per prediction
STRIDE = 5    # one prediction every 5 minutes


def _rolling_slope(x: np.ndarray, w: int) -> np.ndarray:
    """Least-squares slope per minute over a trailing window of w samples."""
    n = len(x)
    k = np.arange(n, dtype=float)
    cx = np.concatenate([[0], np.cumsum(x)])
    ckx = np.concatenate([[0], np.cumsum(k * x)])
    out = np.full(n, np.nan)
    i = np.arange(w - 1, n)
    sx = cx[i + 1] - cx[i + 1 - w]
    skx = ckx[i + 1] - ckx[i + 1 - w]
    st = skx - (i - w + 1) * sx                       # sum of local_t * x
    t_sum = w * (w - 1) / 2.0
    t2_sum = (w - 1) * w * (2 * w - 1) / 6.0
    out[i] = (w * st - t_sum * sx) / (w * t2_sum - t_sum ** 2)
    return out


def window_ends(index: pd.DatetimeIndex) -> np.ndarray:
    return np.arange(WINDOW - 1, len(index), STRIDE)


def _stats(df: pd.DataFrame, ends: np.ndarray, prefix: str = "") -> dict:
    feats = {}
    for c in df.columns:
        x = df[c].to_numpy(dtype=float)
        s = pd.Series(x)
        roll = s.rolling(WINDOW, min_periods=WINDOW)
        feats[f"{prefix}{c}_last"] = x[ends]
        feats[f"{prefix}{c}_mean"] = roll.mean().to_numpy()[ends]
        feats[f"{prefix}{c}_std"] = roll.std().to_numpy()[ends]
        feats[f"{prefix}{c}_min"] = roll.min().to_numpy()[ends]
        feats[f"{prefix}{c}_max"] = roll.max().to_numpy()[ends]
        feats[f"{prefix}{c}_slope60"] = _rolling_slope(x, WINDOW)[ends]
        feats[f"{prefix}{c}_slope15"] = _rolling_slope(x, 15)[ends]
    return feats


def _clock(index: pd.DatetimeIndex, ends: np.ndarray) -> dict:
    h = (index.hour + index.minute / 60.0).to_numpy()[ends]
    return {"tod_sin": np.sin(2 * np.pi * h / 24), "tod_cos": np.cos(2 * np.pi * h / 24)}


def pu_features(profile: Profile, pu: pd.DataFrame, ends: np.ndarray | None = None) -> pd.DataFrame:
    ends = window_ends(pu.index) if ends is None else ends
    feats = _stats(pu, ends)
    for code, a in profile.alarms.items():
        x = pu[a.signal].to_numpy(dtype=float)
        dist = a.distance(x[ends])
        slope = _rolling_slope(x, 15)[ends]
        toward = slope if a.direction == "above" else -slope        # pu per minute toward the limit
        feats[f"{code}_dist"] = dist
        feats[f"{code}_ttl"] = np.where(toward > 1e-6, np.clip(dist / np.maximum(toward, 1e-6), 0, 600), 600.0)
        feats[f"{code}_cond"] = profile.condition_mask(pu, a.condition).to_numpy()[ends].astype(float)
    feats.update(_clock(pu.index, ends))
    return pd.DataFrame(feats, index=pu.index[ends])


def raw_features(raw: pd.DataFrame, ends: np.ndarray | None = None) -> pd.DataFrame:
    ends = window_ends(raw.index) if ends is None else ends
    feats = _stats(raw, ends, prefix="raw_")
    feats.update(_clock(raw.index, ends))
    return pd.DataFrame(feats, index=raw.index[ends])


def sequences(df: pd.DataFrame, ends: np.ndarray) -> np.ndarray:
    """[n_windows, WINDOW, n_channels] float32 array of the trailing WINDOW minutes."""
    x = df.to_numpy(dtype=np.float32)
    offs = np.arange(-WINDOW + 1, 1)
    return x[ends[:, None] + offs[None, :]]


def targets(profile: Profile, labels: pd.DataFrame, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """y[i, a] = alarm a starts within its horizon; m[i, a] = window is valid (alarm not active)."""
    codes = profile.alarm_codes
    y = np.stack([labels[f"{c}_warn"].to_numpy()[ends] for c in codes], axis=1).astype(np.float32)
    m = np.stack([~labels[f"{c}_active"].to_numpy()[ends] for c in codes], axis=1).astype(np.float32)
    return y, m
