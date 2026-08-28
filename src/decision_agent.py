"""
Decision Agent — paper Section II-B-5, Eq. 14-18, Algorithms 1 & 2, Table 3.

This is the deterministic, policy-driven arbitration layer that constitutes
ASPIRE's core contribution. It NEVER learns; every parameter here is a fixed,
calibrated policy constant (see configs/policy_params.yaml). Given identical
inputs it always returns identical decisions (reproducibility/auditability by
construction).
"""
from dataclasses import dataclass, field
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@dataclass
class DecisionAgent:
    cfg: dict

    def sensor_anomaly_intensity(self, anomaly_rate: float, top_k_strength: float,
                                  alpha_s: float = 0.5) -> float:
        # Eq. 14
        return alpha_s * anomaly_rate + (1 - alpha_s) * top_k_strength

    def window_anomaly_intensity(self, wss: float, sev: float, fds: float,
                                  a_wss: float = 0.7, a_sev: float = 0.2,
                                  a_fds: float = 0.1) -> float:
        # Eq. 15-16: a_wss >> a_sev > a_fds, sums to 1 (defaults chosen accordingly;
        # exact calibrated values are not published beyond this ordering constraint)
        return a_wss * sigmoid(abs(wss)) + a_sev * (1 - np.exp(-sev)) + a_fds * sigmoid(fds)

    def detection_risk(self, i_anom_sensor: float, i_anom_window: float) -> float:
        f = self.cfg["detection_fusion"]
        r = f["w_sensor"] * i_anom_sensor + f["w_window"] * i_anom_window  # Eq. 17
        return float(np.clip(r, 0, 1))

    def drift_risk(self, wdi: float, i_drift_sensor: float) -> float:
        # Eq. 18: max(WDI, sensor-drift-intensity) — WDI dominant per Sec II-B-3
        return float(np.clip(max(wdi, i_drift_sensor), 0, 1))

    def decide(self, sensor_agg: dict, window_agent_out: dict,
               pred_probs: np.ndarray) -> dict:
        """
        sensor_agg: output of MasterSensorAggregator.aggregate()
        window_agent_out: output of AdaptiveWindowAgent.step()
        pred_probs: [P_normal, P_warn, P_fault] from the Prediction Agent
        """
        cfg = self.cfg
        p_normal, p_warn, p_fault = [float(p) for p in pred_probs]
        c = int(np.argmax(pred_probs))

        # --- pre-computations ---
        i_anom_s = self.sensor_anomaly_intensity(
            sensor_agg["anomaly_rate"], sensor_agg["top_k_strength"])
        i_anom_w = self.window_anomaly_intensity(
            window_agent_out["WSS"],
            sev=abs(window_agent_out["FDS"]),
            fds=window_agent_out["FDS"])
        i_drift_s = sensor_agg["drift_rate"]  # aggregated sensor drift votes
        wdi = window_agent_out["WDI"]

        r_det = self.detection_risk(i_anom_s, i_anom_w)
        r_drift = self.drift_risk(wdi, i_drift_s)

        det_cfg, drift_cfg = cfg["detection_fusion"], cfg["drift_fusion"]
        fail_cfg, warn_cfg = cfg["failure_prediction"], cfg["warning_gating"]
        lvl_cfg = cfg["warning_level_margins"]

        final_anomaly = r_det >= det_cfg["tau_det"]
        final_drift = r_drift >= drift_cfg["tau_drift"]

        # --- Algorithm 1: Final Failure Logic ---
        final_failure = (p_fault >= fail_cfg["tau_fail"]) or (
            (p_warn >= warn_cfg["tau_warn"])
            and (r_det >= det_cfg["tau_det"])       # detection_risk_high proxy
            and (r_drift >= drift_cfg["tau_drift"])  # drift_risk_high proxy
        )

        # --- Algorithm 1: Final Warning Logic ---
        margin = p_warn - p_normal
        margin_gate = margin >= warn_cfg["delta_margin"]
        transformer_warning = (c == 1) and margin_gate
        near_miss = (c == 0) and (0 < (p_normal - p_warn) < warn_cfg["delta_deficit"])
        warning_evidence = near_miss and (final_drift or final_anomaly)
        final_warning = (transformer_warning or warning_evidence) and (not final_failure)

        # --- Algorithm 1: Warning Level ---
        warning_level = None
        if c == 1:
            if margin >= lvl_cfg["delta_margin_high"]:
                warning_level = "HIGH"
            elif lvl_cfg["delta_margin_low"] <= margin < lvl_cfg["delta_margin_high"]:
                warning_level = "MEDIUM"
            else:
                warning_level = "LOW"
        elif final_warning:  # near-miss recovery path still needs a level for Algorithm 2
            warning_level = "MEDIUM"

        # --- Algorithm 2: Intervention and Alert Logic ---
        if final_failure:
            alert_level, intervention = "CRITICAL", "SEND_ALERT(FIX)"
        elif final_warning and warning_level == "HIGH":
            alert_level, intervention = "HIGH", "SEND_ALERT(PREDICTIVE_INTERVENTION)"
        elif final_warning and warning_level in ("MEDIUM", "LOW"):
            alert_level, intervention = "MEDIUM", "INVOKE_EXPERT_AGENT"
        else:
            alert_level, intervention = "NONE", None

        return {
            "final_anomaly": final_anomaly, "final_drift": final_drift,
            "final_failure": final_failure, "final_warning": final_warning,
            "warning_level": warning_level, "alert_level": alert_level,
            "intervention": intervention,
            "detection_risk": r_det, "drift_risk": r_drift,
            "p_normal": p_normal, "p_warn": p_warn, "p_fault": p_fault,
            "argmax_class": c, "warning_margin": margin,
        }


def decision_agent_from_yaml(policy_params: dict) -> DecisionAgent:
    return DecisionAgent(cfg=policy_params)
