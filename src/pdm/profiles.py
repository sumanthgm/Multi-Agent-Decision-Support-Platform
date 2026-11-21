"""Asset profiles: ratings, signal roles, alarm limits, and the per-unit transform.

A profile is the only thing that changes between asset types and power ratings.
Everything downstream (features, models, agents) works on per-unit (pu) signals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROFILE_DIR = Path(__file__).resolve().parents[2] / "configs" / "profiles"


@dataclass
class Alarm:
    code: str
    name: str
    signal: str          # pu role the limit applies to
    direction: str       # "above" or "below"
    limit_pu: float
    condition: str       # any | day | dg_on | running | full_speed
    warn_minutes: int    # warning horizon H
    ramp_minutes: tuple[int, int]

    def crossed(self, pu: pd.Series) -> pd.Series:
        return pu > self.limit_pu if self.direction == "above" else pu < self.limit_pu

    def distance(self, pu: pd.Series | np.ndarray) -> np.ndarray:
        """Positive = safe margin left before the limit (in pu)."""
        pu = np.asarray(pu, dtype=float)
        return (self.limit_pu - pu) if self.direction == "above" else (pu - self.limit_pu)


@dataclass
class Profile:
    name: str
    asset_type: str
    rating: dict
    signals: dict           # raw column -> role
    alarms: dict[str, Alarm] = field(default_factory=dict)

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, name_or_path: str | Path) -> "Profile":
        p = Path(name_or_path)
        if not p.suffix:
            p = PROFILE_DIR / f"{name_or_path}.yaml"
        d = yaml.safe_load(p.read_text())
        alarms = {
            code: Alarm(code=code, name=a["name"], signal=a["signal"], direction=a["direction"],
                        limit_pu=float(a["limit_pu"]), condition=a["condition"],
                        warn_minutes=int(a["warn_minutes"]), ramp_minutes=tuple(a["ramp_minutes"]))
            for code, a in d["alarms"].items()
        }
        return cls(name=d["name"], asset_type=d["asset_type"], rating=d["rating"],
                   signals=d["signals"], alarms=alarms)

    @property
    def alarm_codes(self) -> list[str]:
        return list(self.alarms)

    # --------------------------------------------------------------- per-unit
    def to_pu(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Map raw columns to roles and divide by the device rating."""
        r = self.rating
        df = raw.rename(columns=self.signals)
        out = pd.DataFrame(index=raw.index)
        out["irradiance"] = df["irradiance"] / 1000.0
        if self.asset_type == "hybrid_inverter":
            out["grid_voltage"] = df["grid_voltage"] / r["v_grid_nominal"]
            out["grid_freq_dev"] = (df["grid_frequency"] / r["f_nominal"] - 1.0).abs()
            out["output_power"] = df["output_power"] / r["p_rated_w"]
            out["pv_power"] = df["pv_power"] / r["p_rated_w"]
            out["pv_voltage"] = df["pv_voltage"] / r["v_pv_max"]
            out["insulation"] = df["insulation"] / r["r_iso_min_kohm"]
            out["battery_voltage"] = df["battery_voltage"] / r["v_bat_nominal"]
            out["battery_soc"] = df["battery_soc"] / 100.0
            out["battery_temp"] = df["battery_temp"] / r["t_bat_trip_c"]
            out["heatsink_temp"] = df["heatsink_temp"] / r["t_hs_trip_c"]
            dg_on = df["dg_on"].fillna(0).astype(float)
            out["dg_on"] = dg_on
            out["dg_freq_dev"] = ((df["dg_frequency"] / r["f_nominal"] - 1.0).abs() * dg_on).fillna(0.0)
        elif self.asset_type == "pump_drive":
            speed = df["speed"] / r["f_nominal"]
            out["dc_voltage"] = df["dc_voltage"] / r["v_dc_nominal"]
            out["output_current"] = df["output_current"] / r["i_rated_a"]
            out["output_power"] = df["output_power"] / r["p_rated_w"]
            out["speed"] = speed
            expected = r["p_rated_w"] * np.clip(speed, 1e-3, None) ** 3
            out["load_ratio"] = np.where(speed > 0.2, df["output_power"] / expected, 1.0)
            out["heatsink_temp"] = df["heatsink_temp"] / r["t_hs_trip_c"]
            out["tank_level"] = df["tank_level"] / 100.0
            out["current_imbalance"] = df["current_imbalance"] / 100.0
            out["flow"] = df["flow"] / r["flow_rated_m3h"]
        else:
            raise ValueError(f"unknown asset_type {self.asset_type}")
        return out

    def pu_scale(self, role: str) -> float | None:
        """Raw units per 1.0 pu for roles that `to_pu` scales linearly; None for derived roles."""
        if role in ("battery_soc", "tank_level", "current_imbalance"):
            return 100.0
        if role == "irradiance":
            return 1000.0
        key = {"grid_voltage": "v_grid_nominal", "output_power": "p_rated_w", "pv_power": "p_rated_w",
               "pv_voltage": "v_pv_max", "insulation": "r_iso_min_kohm", "battery_voltage": "v_bat_nominal",
               "battery_temp": "t_bat_trip_c", "heatsink_temp": "t_hs_trip_c", "dc_voltage": "v_dc_nominal",
               "output_current": "i_rated_a", "flow": "flow_rated_m3h", "speed": "f_nominal"}.get(role)
        return float(self.rating[key]) if key in self.rating else None

    # ------------------------------------------------------------- conditions
    def condition_mask(self, pu: pd.DataFrame, condition: str) -> pd.Series:
        if condition == "any":
            return pd.Series(True, index=pu.index)
        if condition == "day":
            return pu["irradiance"] > 0.05
        if condition == "dg_on":
            return pu["dg_on"] > 0.5
        if condition == "running":
            return pu["speed"] > 0.2
        if condition == "full_speed":
            return pu["speed"] > 0.80
        raise ValueError(condition)


def list_profiles() -> list[str]:
    return sorted(p.stem for p in PROFILE_DIR.glob("*.yaml"))
