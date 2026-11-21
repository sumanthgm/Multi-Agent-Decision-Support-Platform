"""Healthy telemetry for a hybrid inverter (PV + battery + grid + generator) of any rating."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..profiles import Profile
from .base import bump, lag


def healthy_inverter(profile: Profile, weather: pd.DataFrame, seed: int) -> pd.DataFrame:
    r = profile.rating
    rng = np.random.default_rng(seed + 101)
    n = len(weather)
    days = n // 1440
    h = weather["hour"].to_numpy()
    day = np.arange(n) // 1440
    g = weather["g"].to_numpy()
    t_amb = weather["t_amb"].to_numpy()
    P = r["p_rated_w"]

    t_mod = t_amb + 0.03 * g
    p_pv = np.clip(1.15 * P * g / 1000 * (1 - 0.004 * (t_mod - 25)), 0, P) * (1 + rng.normal(0, 0.01, n))

    shape = 0.12 + 0.25 * bump(h, 7.5, 1.0) + 0.45 * bump(h, 19.5, 1.5) + 0.10 * bump(h, 13, 2.0)
    load_pu = shape * rng.uniform(0.8, 1.1, days)[day] + rng.normal(0, 0.015, n)
    for _ in range(days * 4):                                  # appliance switch-ons
        s = rng.integers(0, n - 30)
        load_pu[s:s + rng.integers(5, 20)] += rng.uniform(0.10, 0.25)
    p_out = np.clip(load_pu, 0.03, 0.90) * P

    cap_wh = r["battery_kwh"] * 1000.0
    soc = np.empty(n)
    soc[0] = 60.0
    net = p_pv - p_out
    for t in range(1, n):
        soc[t] = np.clip(soc[t - 1] + net[t] / 60.0 / cap_wh * 100.0 * 0.95, 15.0, 100.0)
    export_pu = np.where(soc >= 99.5, np.clip(net, 0, None), 0) / P

    vb = r["v_bat_nominal"]
    v_bat = vb * (0.94 + 0.12 * soc / 100 + 0.02 * net / P) + rng.normal(0, 0.002 * vb, n)
    t_bat = lag(t_amb + 3 + 6 * np.abs(net) / P, 20) + rng.normal(0, 0.2, n)
    t_hs = lag(t_amb + 30 * np.maximum(p_pv, p_out) / P, 10) + rng.normal(0, 0.3, n)

    vg = r["v_grid_nominal"]
    ac_v = vg * (1 + 0.015 * np.sin(2 * np.pi * (h - 14) / 24) + 0.02 * export_pu + rng.normal(0, 0.004, n))
    fn = r["f_nominal"]
    f_ar = np.zeros(n)
    e = rng.normal(0, 0.00012, n)
    for t in range(1, n):
        f_ar[t] = 0.95 * f_ar[t - 1] + e[t]
    ac_f = fn * (1 + f_ar)

    v_pv = np.where(g > 20, r["v_pv_max"] * 0.72 * (1 - 0.0035 * (t_mod - 25)), 0.0) + rng.normal(0, 1.0, n) * (g > 20)
    dawn = bump(h, 6.0, 1.2)
    r_iso = 1500 * (1 - 0.4 * dawn) * (1 + rng.normal(0, 0.03, n))

    dg_on = np.zeros(n)
    for d in range(days):
        if rng.random() < 0.35:
            s = d * 1440 + int(rng.uniform(18.0, 19.0) * 60)
            dg_on[s:s + int(rng.uniform(90, 150))] = 1
    dg_f = np.where(dg_on > 0, fn * (1 + rng.normal(0, 0.004, n)), fn)

    return pd.DataFrame({
        "ac_v": ac_v, "ac_f": ac_f, "p_out": p_out, "p_pv": p_pv, "v_pv": v_pv, "r_iso": r_iso,
        "v_bat": v_bat, "soc": soc, "t_bat": t_bat, "t_hs": t_hs, "dg_f": dg_f, "dg_on": dg_on, "g": g,
    }, index=weather.index)
