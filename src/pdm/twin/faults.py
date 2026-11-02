"""Fault injection: each alarm's early sign grows as a ramp until it crosses (or nearly crosses) its limit.

- True episodes overshoot the limit, then hold and recover (a repair).
- Near-miss episodes stop short of the limit; they are hard negatives so the model cannot
  just learn "any ramp means alarm".
Onset, ramp length, ramp curvature, overshoot and hold time are random per seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..profiles import Profile

INV_MAP = {  # pu role -> (raw column, rating key used as scale)
    "grid_voltage": ("ac_v", "v_grid_nominal"),
    "grid_freq_dev": ("ac_f", "f_nominal"),
    "pv_voltage": ("v_pv", "v_pv_max"),
    "insulation": ("r_iso", "r_iso_min_kohm"),
    "battery_voltage": ("v_bat", "v_bat_nominal"),
    "battery_temp": ("t_bat", "t_bat_trip_c"),
    "output_power": ("p_out", "p_rated_w"),
    "heatsink_temp": ("t_hs", "t_hs_trip_c"),
    "dg_freq_dev": ("dg_f", "f_nominal"),
}
DRV_MAP = {
    "dc_voltage": ("v_dc", "v_dc_nominal"),
    "output_current": ("i_out", "i_rated_a"),
    "heatsink_temp": ("t_hs", "t_hs_trip_c"),
    "tank_level": ("level", None),
    "current_imbalance": ("i_imb", None),
    "load_ratio": (None, None),
}


def _shape(n_ramp: int, n_hold: int, n_rec: int, power: float) -> np.ndarray:
    ramp = (np.arange(1, n_ramp + 1) / n_ramp) ** power
    return np.concatenate([ramp, np.ones(n_hold), np.linspace(1, 0, n_rec)])


def inject_faults(
    profile: Profile, raw: pd.DataFrame, seed: int, n_true: int = 8, n_near: int = 4, gap_min: int = 120
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed + 303)
    raw = raw.copy()
    base_pu = profile.to_pu(raw)
    n = len(raw)
    busy = np.zeros(n, dtype=bool)
    rmap = INV_MAP if profile.asset_type == "hybrid_inverter" else DRV_MAP
    episodes = []

    plan = [(code, kind) for code in profile.alarm_codes for kind in ["true"] * n_true + ["near"] * n_near]
    rng.shuffle(plan)
    for code, kind in plan:
        a = profile.alarms[code]
        cond = profile.condition_mask(base_pu, a.condition).to_numpy()
        for _ in range(800):
            d_ramp = int(rng.integers(a.ramp_minutes[0], a.ramp_minutes[1] + 1))
            d_hold = int(rng.integers(10, 30 if a.condition in ("dg_on", "running", "full_speed") else 60))
            d_rec = 20
            t0 = int(rng.integers(60, n - d_ramp - d_hold - d_rec - 60))
            t_cross, t_end = t0 + d_ramp, t0 + d_ramp + d_hold
            span = slice(max(0, t0 - gap_min), min(n, t_end + d_rec + gap_min))
            if busy[span].any() or not cond[t_cross - 5 : t_end].all():
                continue
            break
        else:
            continue  # no slot found for this episode
        busy[t0 : t_end + d_rec] = True
        idx = slice(t0, t_end + d_rec)
        b = float(base_pu[a.signal].iloc[t_cross - 10 : t_cross].mean())
        L = a.limit_pu
        sign = 1.0 if a.direction == "above" else -1.0
        if kind == "true":
            target = L * (1 + sign * rng.uniform(0.02, 0.08))  # overshoot relative to the limit
        else:
            target = b + rng.uniform(0.35, 0.80) * (L - b)
        delta = (target - b) * _shape(d_ramp, d_hold, d_rec, rng.uniform(1.0, 2.5))
        _apply(profile, raw, rmap, a.signal, idx, delta, rng, b)
        episodes.append(
            dict(
                code=code,
                kind=kind,
                start=raw.index[t0],
                cross=raw.index[t_cross],
                end=raw.index[t_end],
                base_pu=b,
                target_pu=target,
            )
        )
    for c in raw.columns:  # physical quantities stay non-negative
        if c not in ("ac_f", "dg_f"):
            raw[c] = raw[c].clip(lower=0)
    ep = pd.DataFrame(episodes).sort_values("start").reset_index(drop=True) if episodes else pd.DataFrame()
    return raw, ep


def _apply(profile, raw, rmap, signal, idx, delta, rng, b_pu):
    r = profile.rating
    col, key = rmap[signal]
    if signal in ("grid_freq_dev", "dg_freq_dev"):
        s = rng.choice([-1.0, 1.0])
        raw.iloc[idx, raw.columns.get_loc(col)] += s * delta * r[key]
    elif signal == "load_ratio":  # dry run / underload: power and flow fall at the same speed
        factor = 1.0 + delta
        for c, w in (("p_out", 1.0), ("flow", 1.0), ("i_out", 0.6)):
            j = raw.columns.get_loc(c)
            raw.iloc[idx, j] = raw.iloc[idx, j].to_numpy() * (1 - w + w * factor)
    elif signal == "insulation":  # insulation resistance falls multiplicatively (moisture, ageing)
        j = raw.columns.get_loc(col)
        seg = raw.iloc[idx, j].to_numpy()
        factor = 1.0 + delta / max(b_pu, 1e-6)
        raw.iloc[idx, j] = seg * np.clip(factor, 0.01, None)
    elif signal in ("tank_level", "current_imbalance"):
        j = raw.columns.get_loc(col)
        raw.iloc[idx, j] = np.clip(raw.iloc[idx, j].to_numpy() + delta * 100.0, 0, 100)
    else:
        j = raw.columns.get_loc(col)
        raw.iloc[idx, j] = raw.iloc[idx, j].to_numpy() + delta * r[key]
        # physically linked side effects
        if signal == "output_power":
            k = raw.columns.get_loc("t_hs")
            raw.iloc[idx, k] = raw.iloc[idx, k].to_numpy() + 30.0 * delta  # degC per pu power, as in the healthy model
        if signal == "output_current" and "t_hs" in raw:
            k = raw.columns.get_loc("t_hs")
            raw.iloc[idx, k] = raw.iloc[idx, k].to_numpy() + 10 * delta
