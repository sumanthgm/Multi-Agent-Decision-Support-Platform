"""Orchestrator: runs the four agents on the latest telemetry window for one asset.

raw window -> profile.to_pu -> Monitor (live alarms) -> Early-Warning (alarms within H)
           -> Forecast (minutes to trip for the warned signal) -> Advisor (cited checklist)

The flow is a fixed graph, written in plain Python so it runs anywhere; it maps one-to-one
onto a LangGraph StateGraph if you want an agent-framework implementation.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ..features import WINDOW, pu_features
from ..profiles import Profile
from .advisor import AdvisorAgent
from .early_warning import EarlyWarningXGB
from .forecast import ForecastAgent
from .monitor import MonitorAgent


class Platform:
    def __init__(
        self,
        profile: Profile,
        model: EarlyWarningXGB,
        forecast_method: str = "trend",
        advisor: AdvisorAgent | None = None,
    ):
        self.profile = profile
        self.monitor = MonitorAgent(profile)
        self.model = model
        self.forecaster = ForecastAgent(forecast_method)
        self.advisor = advisor or AdvisorAgent()

    @classmethod
    def from_files(cls, profile_name: str, model_path: str | Path, **kw) -> Platform:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return cls(Profile.load(profile_name), model, **kw)

    def step(self, raw_window: pd.DataFrame) -> dict:
        if len(raw_window) < WINDOW:
            return {"status": "warming_up", "rows": len(raw_window)}
        pu = self.profile.to_pu(raw_window)
        alarms = self.monitor.active(pu)
        X = pu_features(self.profile, pu, np.array([len(pu) - 1]))
        prob = self.model.predict_proba(X)[0]
        warnings = []
        for j, code in enumerate(self.model.codes):
            if prob[j] >= self.model.thresholds[code] and code not in {a["code"] for a in alarms}:
                a = self.profile.alarms[code]
                ttt = self.forecaster.time_to_trip(pu[a.signal].to_numpy(), a)
                warnings.append(
                    {
                        "code": code,
                        "name": a.name,
                        "probability": float(prob[j]),
                        "horizon_min": a.warn_minutes,
                        "forecast_minutes_to_trip": ttt,
                    }
                )
        warnings.sort(key=lambda w: -w["probability"])
        top = alarms[0]["code"] if alarms else (warnings[0]["code"] if warnings else None)
        return {
            "time": str(raw_window.index[-1]),
            "asset": self.profile.name,
            "state": "ALARM" if alarms else ("WARNING" if warnings else "NORMAL"),
            "alarms": alarms,
            "warnings": warnings,
            "advice": self.advisor.advise(top) if top else None,
        }
