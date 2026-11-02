"""Shared weather for the telemetry twin: irradiance and ambient temperature at 1-minute steps.

The irradiance is synthetic (clear-sky bell shape with day-to-day and minute-to-minute
clouds). To use real weather, pass a DataFrame with columns `g` (W/m2) and `t_amb` (degC)
from an open dataset such as the La Reunion PV dataset (see data/real/README.md).
"""
from __future__ import annotations

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

    cloudiness = rng.beta(2.0, 5.0, size=days)[day]          # how cloudy each day is
    ar = np.zeros(n)
    eps = rng.normal(0, 0.35, size=n)
    for t in range(1, n):                                     # fast passing clouds
        ar[t] = 0.97 * ar[t - 1] + eps[t]
    passing = 1.0 / (1.0 + np.exp(-ar))
    k = np.clip(1.0 - cloudiness * (0.4 + 0.9 * passing), 0.08, 1.0)
    g = np.clip(clear * k + rng.normal(0, 8, n) * (clear > 0), 0, None)

    t_amb = (26 + 5 * np.sin(2 * np.pi * (h - 9) / 24) + rng.normal(0, 1.5, days)[day]
             + rng.normal(0, 0.3, n))
    return pd.DataFrame({"g": g, "t_amb": t_amb, "hour": h}, index=idx)


def lag(x: np.ndarray, tau_min: float) -> np.ndarray:
    """First-order thermal lag with time constant tau (minutes)."""
    a = 1.0 / tau_min
    y = np.empty_like(x)
    y[0] = x[0]
    for t in range(1, len(x)):
        y[t] = y[t - 1] + a * (x[t] - y[t - 1])
    return y


bump = _bump
