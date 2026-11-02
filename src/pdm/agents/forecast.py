"""Forecast Agent: forecasts a per-unit signal and estimates minutes until it crosses an alarm limit.

Methods: persistence, seasonal naive (same time yesterday), linear trend (last 30 min), and
Chronos-Bolt zero-shot (optional; needs `pip install chronos-forecasting` and model download).
"""

from __future__ import annotations

import numpy as np

from ..profiles import Alarm

_CHRONOS = None


def _chronos():
    global _CHRONOS
    if _CHRONOS is None:
        import torch
        from chronos import BaseChronosPipeline

        _CHRONOS = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-small", device_map="cpu", torch_dtype=torch.float32
        )
    return _CHRONOS


def forecast(history: np.ndarray, horizon: int, method: str = "trend") -> np.ndarray:
    history = np.asarray(history, dtype=float)
    if method == "persistence":
        return np.repeat(history[-1], horizon)
    if method == "seasonal":
        if len(history) >= 1440 + horizon:
            return history[-1440 : -1440 + horizon] + (history[-1] - history[-1440 - 1])
        return np.repeat(history[-1], horizon)
    if method == "trend":
        w = min(30, len(history))
        t = np.arange(w)
        slope, icpt = np.polyfit(t, history[-w:], 1)
        return icpt + slope * (w - 1 + np.arange(1, horizon + 1))
    if method == "chronos":
        import torch

        ctx = torch.tensor(history[-512:], dtype=torch.float32)
        q, mean = _chronos().predict_quantiles(context=[ctx], prediction_length=horizon, quantile_levels=[0.5])
        return mean[0].numpy()
    raise ValueError(method)


def minutes_to_limit(path: np.ndarray, alarm: Alarm) -> int | None:
    """First forecast step (minutes ahead) at which the alarm limit is crossed, or None."""
    path = np.asarray(path, dtype=float)
    crossed = path > alarm.limit_pu if alarm.direction == "above" else path < alarm.limit_pu
    idx = np.flatnonzero(crossed)
    return int(idx[0]) + 1 if len(idx) else None


class ForecastAgent:
    def __init__(self, method: str = "trend", horizon: int = 60):
        self.method, self.horizon = method, horizon

    def time_to_trip(self, pu_history, alarm: Alarm) -> int | None:
        return minutes_to_limit(forecast(pu_history, self.horizon, self.method), alarm)
