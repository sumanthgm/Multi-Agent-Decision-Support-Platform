"""Healthy telemetry for a solar water-pump drive (VFD + motor + pump + tank) of any rating.

No open solar-pump telemetry dataset was found, so this is generated from irradiance with
pump affinity laws (flow ~ speed, power ~ speed^3) and a simple tank level controller.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..profiles import Profile
from .base import bump, lag


def healthy_drive(profile: Profile, weather: pd.DataFrame, seed: int) -> pd.DataFrame:
    r = profile.rating
    rng = np.random.default_rng(seed + 202)
    n = len(weather)
    h = weather["hour"].to_numpy()
    g = weather["g"].to_numpy()
    t_amb = weather["t_amb"].to_numpy()
    P = r["p_rated_w"]

    t_mod = t_amb + 0.03 * g
    p_avail = 1.3 * P * g / 1000 * (1 - 0.004 * (t_mod - 25))
    sun_ok = p_avail > 0.18 * P

    q_rated = r["flow_rated_m3h"]
    use = q_rated * 0.55 * (bump(h, 7.0, 1.5) + bump(h, 18.0, 1.5)) + 0.05 * q_rated
    level = np.empty(n)
    speed = np.zeros(n)
    level[0] = 50.0
    inhibit = False
    vol = r["tank_m3"]
    for t in range(1, n):
        if level[t - 1] > 88:
            inhibit = True
        elif level[t - 1] < 75:
            inhibit = False
        run = sun_ok[t] and not inhibit
        speed[t] = np.clip((p_avail[t] / P) ** (1 / 3), 0.45, 1.0) if run else 0.0
        q = q_rated * max(0.0, (speed[t] - 0.4) / 0.6)
        level[t] = np.clip(level[t - 1] + (q - use[t]) / 60.0 / vol * 100.0, 2.0, 100.0)

    running = speed > 0
    p_out = np.where(running, P * speed ** 3 * (1 + rng.normal(0, 0.02, n)), 0.0)
    i_out = np.where(running, r["i_rated_a"] * (0.25 + 0.65 * p_out / P), 0.0) * (1 + rng.normal(0, 0.01, n))
    v_dc = np.where(running, r["v_dc_nominal"] * (0.93 + 0.07 * np.minimum(g / 1000, 1)), 0.0) * (1 + rng.normal(0, 0.005, n))
    t_hs = lag(t_amb + 25 * p_out / P, 10) + rng.normal(0, 0.3, n)
    flow = np.where(running, q_rated * np.clip((speed - 0.4) / 0.6, 0, None), 0.0)
    i_imb = np.where(running, 2.0 + rng.normal(0, 0.5, n), 0.0)

    return pd.DataFrame({
        "v_dc": v_dc, "i_out": i_out, "p_out": p_out, "f_out": speed * r["f_nominal"], "t_hs": t_hs,
        "level": level, "i_imb": i_imb, "flow": flow, "g": g,
    }, index=weather.index)
