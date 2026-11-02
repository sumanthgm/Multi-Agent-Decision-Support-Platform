"""FastAPI service: ingest live telemetry rows and return the platform's current decision.

Run:  uvicorn pdm.serve.api:app --port 8000      (from the repo root, with `pip install -e .`)
Env:  PDM_PROFILE (profile used when a request names none; default inverter_hybrid_5kw),
      PDM_MODEL (model file for PDM_PROFILE's asset type; default results/models/<profile>_ours.pkl),
      OLLAMA_URL / OLLAMA_MODEL (optional LLM wording for /advice).

Every endpoint takes `?profile=<name>`, so one service watches inverters and pump drives side by side.
`POST /feed/start` replays twin telemetry inside the service, so the dashboard has live data without
running scripts/replay.py.
"""

from __future__ import annotations

import os
import pickle
import threading
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..agents.advisor import AdvisorAgent
from ..agents.forecast import forecast, minutes_to_limit
from ..agents.orchestrator import Platform
from ..profiles import Alarm, Profile, list_profiles
from ..twin import generate

ROOT = Path(__file__).resolve().parents[3]
PROFILE = os.environ.get("PDM_PROFILE", "inverter_hybrid_5kw")
PREFILL = 240  # minutes of history loaded at once when a feed starts


def _train_profile(profile: Profile) -> str:
    """Model files are trained on the 5 kW / 5 HP unit; other ratings reuse them through their profile."""
    return "inverter_hybrid_5kw" if profile.asset_type == "hybrid_inverter" else "pump_drive_5hp"


def load_platform(profile_name: str = PROFILE) -> Platform:
    profile = Profile.load(profile_name)
    path = ROOT / "results" / "models" / f"{_train_profile(profile)}_ours.pkl"
    if os.environ.get("PDM_MODEL") and _train_profile(profile) == _train_profile(Profile.load(PROFILE)):
        path = Path(os.environ["PDM_MODEL"])
    from ..training import train_and_save

    if not path.exists():  # first start: train the model for this asset type
        path = train_and_save(_train_profile(profile))
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
    except Exception:  # pickled by another xgboost version: retrain with the installed one
        with open(train_and_save(_train_profile(profile), out_dir=path.parent), "rb") as f:
            model = pickle.load(f)
    # /status is polled every few seconds, so its advice skips the LLM; /advice asks it on demand
    return Platform(profile, model, advisor=AdvisorAgent(llm=False))


class Reading(BaseModel):
    timestamp: str
    values: dict[str, float]


app = FastAPI(title="Agentic PdM platform")
_assets: dict[str, dict] = {}
_assets_lock = threading.Lock()
_advisor = AdvisorAgent()


def _asset(name: str) -> dict:
    if name not in list_profiles():
        raise HTTPException(404, f"unknown profile {name}")
    with _assets_lock:
        if name not in _assets:
            _assets[name] = {
                "profile": Profile.load(name),
                "platform": None,
                "buffer": deque(maxlen=1440),
                "lock": threading.Lock(),
                "model_lock": threading.Lock(),
                "feed": None,
            }
        return _assets[name]


def _platform(a: dict) -> Platform:
    with a["model_lock"]:
        if a["platform"] is None:
            a["platform"] = load_platform(a["profile"].name)
        return a["platform"]


def _append(a: dict, ts, values: dict) -> None:
    with a["lock"]:
        a["buffer"].append({"timestamp": pd.Timestamp(ts), **values})


def _window(a: dict) -> pd.DataFrame:
    with a["lock"]:
        rows = list(a["buffer"])
    return pd.DataFrame(rows).set_index("timestamp") if rows else pd.DataFrame()


def _raw_limit(p: Profile, al: Alarm) -> dict:
    """The alarm limit in the raw signal's engineering units, where the pu transform can be inverted."""
    col = {role: c for c, role in p.signals.items()}
    scale = p.pu_scale(al.signal)
    if scale is not None:
        return {"raw_column": col.get(al.signal), "raw_limits": [al.limit_pu * scale]}
    if al.signal in ("grid_freq_dev", "dg_freq_dev"):  # |f / f_nominal - 1| > limit  ->  a band around f_nominal
        f = p.rating["f_nominal"]
        src = "grid_frequency" if al.signal == "grid_freq_dev" else "dg_frequency"
        return {"raw_column": col.get(src), "raw_limits": [f * (1 - al.limit_pu), f * (1 + al.limit_pu)]}
    return {"raw_column": None, "raw_limits": []}


def _feed_info(a: dict) -> dict:
    f = a["feed"]
    return {k: v for k, v in f.items() if k != "stop"} if f else {"running": False}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "profile": PROFILE,
        "rows": len(_assets[PROFILE]["buffer"]) if PROFILE in _assets else 0,
        "llm": bool(os.environ.get("OLLAMA_URL")),
        "assets": {n: {"rows": len(a["buffer"]), "feed": _feed_info(a)} for n, a in _assets.items()},
    }


@app.get("/profiles")
def profiles():
    out = []
    for name in list_profiles():
        p = Profile.load(name)
        alarms = [
            {
                "code": c,
                "name": al.name,
                "signal": al.signal,
                "direction": al.direction,
                "limit_pu": al.limit_pu,
                "condition": al.condition,
                "warn_minutes": al.warn_minutes,
                **_raw_limit(p, al),
            }
            for c, al in p.alarms.items()
        ]
        out.append(
            {"name": name, "asset_type": p.asset_type, "rating": p.rating, "signals": p.signals, "alarms": alarms}
        )
    return out


@app.post("/ingest")
def ingest(r: Reading, profile: str = PROFILE):
    a = _asset(profile)
    _append(a, r.timestamp, r.values)
    return {"rows": len(a["buffer"])}


@app.get("/status")
def status(profile: str = PROFILE):
    a = _asset(profile)
    df = _window(a)
    if df.empty:
        return {"status": "no data", "asset": profile, "feed": _feed_info(a)}
    return {**_platform(a).step(df), "rows": len(df), "feed": _feed_info(a)}


@app.get("/history")
def history(n: int = 240, profile: str = PROFILE):
    a = _asset(profile)
    with a["lock"]:
        rows = list(a["buffer"])[-n:]
    return [{**r, "timestamp": str(r["timestamp"])} for r in rows]


@app.get("/forecast")
def forecast_view(n: int = 120, profile: str = PROFILE):
    """Per alarm: recent pu history, the Forecast Agent's path, the limit and minutes to trip."""
    a = _asset(profile)
    df = _window(a)
    if len(df) < 2:
        return {"times": [], "horizon": 0, "alarms": []}
    p, fc = a["profile"], _platform(a).forecaster
    pu = p.to_pu(df)
    now = pu.tail(1)
    out = []
    for code, al in p.alarms.items():
        s = pu[al.signal].to_numpy(dtype=float)
        path = forecast(s, fc.horizon, fc.method)
        out.append(
            {
                "code": code,
                "name": al.name,
                "signal": al.signal,
                "direction": al.direction,
                "condition": al.condition,
                "limit_pu": al.limit_pu,
                "value_pu": float(s[-1]),
                "margin_pu": float(al.distance(s[-1:])[0]),
                "condition_met": bool(p.condition_mask(now, al.condition).iloc[0]),
                "minutes_to_trip": minutes_to_limit(path, al),
                "scale": p.pu_scale(al.signal),
                "history_pu": s[-n:].tolist(),
                "forecast_pu": path.tolist(),
            }
        )
    return {"times": [str(t) for t in pu.index[-n:]], "horizon": fc.horizon, "alarms": out}


def _context(a: dict, code: str) -> str:
    """Live readings handed to the LLM so it can rank the manual's checks for this situation."""
    p, df = a["profile"], _window(a)
    if df.empty:
        return f"Asset {p.name} ({p.asset_type}); no live data."
    lines = [f"Asset {p.name} ({p.asset_type}) at {df.index[-1]}."]
    al = p.alarms.get(code)
    if al:
        s = p.to_pu(df)[al.signal].to_numpy(dtype=float)
        ttt = _platform(a).forecaster.time_to_trip(s, al)
        lines.append(
            f"{code} {al.name}: {al.signal} = {s[-1]:.3f} pu, trips {al.direction} {al.limit_pu} pu; "
            + (f"forecast to cross in {ttt} min." if ttt else "no crossing forecast in the next hour.")
        )
    lines.append("Latest readings: " + ", ".join(f"{k}={v:.4g}" for k, v in df.iloc[-1].items()))
    return " ".join(lines)


@app.get("/advice")
def advice(code: str, profile: str = PROFILE, llm: bool = False):
    a = _asset(profile)
    context = _context(a, code)
    return {**_advisor.advise(code, context, llm=llm), "context": context}


def _feed_loop(a: dict, raw: pd.DataFrame, feed: dict) -> None:
    for ts, row in raw.iterrows():
        _append(a, ts, row.to_dict())
        if feed["stop"].wait(1.0 / feed["speed"]):
            break
    feed["running"] = False


@app.post("/feed/start")
def feed_start(profile: str = PROFILE, scenario: str = "fault", speed: float = 2.0, seed: int | None = None):
    """Stream twin telemetry into this profile's buffer at `speed` rows (minutes) per second.

    scenario=fault starts 45 minutes before the first alarm onset, so the early warning, forecast
    and alarm all show within a minute at the default speed; scenario=normal starts at 06:00.
    """
    a = _asset(profile)
    feed_stop(profile)
    seed = int(np.random.default_rng().integers(1000)) if seed is None else seed
    d = generate(a["profile"], days=3, seed=seed, n_true=3, n_near=1)
    raw, labels = d["raw"], d["labels"]
    start = 360
    if scenario == "fault":
        onsets = sorted(
            raw.index.get_loc(t) - 45 for c in a["profile"].alarm_codes for t in labels.index[labels[f"{c}_onset"] > 0]
        )
        start = next((i for i in onsets if i >= PREFILL), start)
    start = max(start, PREFILL)
    with a["lock"]:
        a["buffer"].clear()
    for ts, row in raw.iloc[start - PREFILL : start].iterrows():
        _append(a, ts, row.to_dict())
    feed = {"running": True, "scenario": scenario, "speed": max(speed, 0.1), "seed": seed, "stop": threading.Event()}
    a["feed"] = feed
    threading.Thread(target=_feed_loop, args=(a, raw.iloc[start:], feed), daemon=True).start()
    return _feed_info(a)


@app.post("/feed/stop")
def feed_stop(profile: str = PROFILE):
    a = _asset(profile)
    if a["feed"]:
        a["feed"]["stop"].set()
        a["feed"]["running"] = False
    return _feed_info(a)
