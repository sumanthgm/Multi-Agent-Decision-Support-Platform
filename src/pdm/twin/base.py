"""Shared weather for the telemetry twin: irradiance and ambient temperature at 1-minute steps.

The irradiance is synthetic (clear-sky bell shape with day-to-day and minute-to-minute
clouds). To use real weather, pass a DataFrame with columns `g` (W/m2) and `t_amb` (degC)
from an open dataset such as the La Reunion PV dataset (see data/real/README.md).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _bump(hours: np.ndarray, centre: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((hours - centre) / width) ** 2)


def make_weather(days: int, seed: int, start: str = "2026-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = days * 1440
    idx = pd.date_range(start, periods=n, freq="1min")
    h = (np.arange(n) % 1440) / 60.0
    day = np.arange(n) // 1440

    clear = np.clip(np.sin(np.pi * (h - 6.25) / 12.0), 0, None) ** 1.3 * 1000.0

    cloudiness = rng.beta(2.0, 5.0, size=days)[day]  # how cloudy each day is
    ar = np.zeros(n)
    eps = rng.normal(0, 0.35, size=n)
    for t in range(1, n):  # fast passing clouds
        ar[t] = 0.97 * ar[t - 1] + eps[t]
    passing = 1.0 / (1.0 + np.exp(-ar))
    k = np.clip(1.0 - cloudiness * (0.4 + 0.9 * passing), 0.08, 1.0)
    g = np.clip(clear * k + rng.normal(0, 8, n) * (clear > 0), 0, None)

    t_amb = 26 + 5 * np.sin(2 * np.pi * (h - 9) / 24) + rng.normal(0, 1.5, days)[day] + rng.normal(0, 0.3, n)
    return pd.DataFrame({"g": g, "t_amb": t_amb, "hour": h}, index=idx)


def load_weather(path: str | Path, max_gap_min: int = 30) -> pd.DataFrame:
    """Measured 1-minute weather (columns time, g, t_amb; see scripts/prepare_reunion_weather.py).

    Gaps up to `max_gap_min` are interpolated; days that still have missing minutes are dropped,
    so every remaining day has 1440 rows, as the healthy-telemetry models expect.
    """
    w = pd.read_csv(path, parse_dates=["time"], index_col="time")
    first, last = w.index[0].normalize(), w.index[-1].normalize() + pd.Timedelta("1D")
    w = w.reindex(pd.date_range(first, last, freq="1min", inclusive="left"))
    w = w.interpolate(limit=max_gap_min, limit_area="inside")
    w = w[w.notna().all(axis=1).groupby(w.index.normalize()).transform("all")]
    w["hour"] = w.index.hour + w.index.minute / 60.0
    return w


def weather_block(w: pd.DataFrame, start_day: int, days: int) -> pd.DataFrame:
    """`days` complete measured days from day number `start_day`, on one continuous 1-minute index.

    Dropped days leave jumps at midnight, when PV output is zero, so the days are joined end to end.
    """
    day_ids = w.index.normalize().unique()
    if start_day + days > len(day_ids):
        raise ValueError(
            f"measured weather has {len(day_ids)} complete days; days {start_day}..{start_day + days} asked"
        )
    b = w[w.index.normalize().isin(day_ids[start_day : start_day + days])].copy()
    b.index = pd.date_range(day_ids[start_day], periods=len(b), freq="1min")
    return b


def lag(x: np.ndarray, tau_min: float) -> np.ndarray:
    """First-order thermal lag with time constant tau (minutes)."""
    a = 1.0 / tau_min
    y = np.empty_like(x)
    y[0] = x[0]
    for t in range(1, len(x)):
        y[t] = y[t - 1] + a * (x[t] - y[t - 1])
    return y


bump = _bump
