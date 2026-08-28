"""
Ablation study — paper Section IV-E, Table 11.

Six variants evaluated over the SAME cached perception outputs (no retraining):
  V0: Transformer baseline (argmax only) — no Decision Agent at all.
  V1: Full ASPIRE (reported system).
  V2: No sensor agents (drop sensor evidence from detection/drift risk).
  V3: No window agent (drop WSS/FDS/WDI contextual evidence).
  V4: No margin gate (delta_margin = 0, i.e. warning fires whenever c==1).
  V5: No near-miss recovery (delta_deficit = 0).
  V6: Post-processing baseline — margin gate ONLY, no sensor/window context at all
      (the "strongest possible single-layer post-processing comparison").
"""
import copy
import numpy as np
from decision_agent import DecisionAgent


def make_variant_cfg(base_cfg: dict, variant: str) -> dict:
    cfg = copy.deepcopy(base_cfg)
    if variant == "V4_no_margin_gate":
        cfg["warning_gating"]["delta_margin"] = 0.0
    elif variant == "V5_no_near_miss":
        cfg["warning_gating"]["delta_deficit"] = 0.0
    elif variant == "V6_post_processing_baseline":
        cfg["warning_gating"]["delta_deficit"] = 0.0
        # V6 additionally zeroes sensor/window contribution to risk fusion
        cfg["detection_fusion"]["w_sensor"] = 0.0
        cfg["detection_fusion"]["w_window"] = 0.0
        cfg["drift_fusion"]["w_wdi"] = 0.0
        cfg["drift_fusion"]["w_sensor_drift"] = 0.0
        cfg["drift_fusion"]["w_fdi"] = 0.0
    return cfg


def run_variant(variant: str, base_cfg: dict, sensor_agg_list, window_out_list,
                 pred_probs_list, zero_sensor: bool = False, zero_window: bool = False):
    cfg = make_variant_cfg(base_cfg, variant)
    agent = DecisionAgent(cfg=cfg)
    decisions = []
    for sensor_agg, window_out, probs in zip(sensor_agg_list, window_out_list, pred_probs_list):
        sa = dict(sensor_agg)
        wo = dict(window_out)
        if zero_sensor or variant == "V2_no_sensor_agents":
            sa["anomaly_rate"] = 0.0
            sa["top_k_strength"] = 0.0
            sa["drift_rate"] = 0.0
        if zero_window or variant == "V3_no_window_agent":
            wo["WSS"] = 0.0
            wo["FDS"] = 0.0
            wo["WDI"] = 0.0
        out = agent.decide(sa, wo, probs)
        cls = 2 if out["final_failure"] else (1 if out["final_warning"] else 0)
        decisions.append(cls)
    return np.array(decisions)


VARIANTS = [
    "V1_full_aspire",
    "V2_no_sensor_agents",
    "V3_no_window_agent",
    "V4_no_margin_gate",
    "V5_no_near_miss",
    "V6_post_processing_baseline",
]
