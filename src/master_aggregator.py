"""
Master Sensor Aggregator Agent — paper Section II-B-2.

Deterministic, no trainable parameters. Aggregates the boolean outputs
(is_anomaly, drift_flag, needs_retrain) across all S Sensor Agents into
system-level fractions and flags via fixed thresholds.
"""
import numpy as np


class MasterSensorAggregator:
    def __init__(self, cfg: dict):
        self.system_anomaly_threshold = cfg.get("system_anomaly_threshold", 0.30)
        self.system_drift_threshold = cfg.get("system_drift_threshold", 0.20)
        self.system_needs_retrain_threshold = cfg.get("system_needs_retrain_threshold", 0.15)

    def aggregate(self, sensor_outputs: list[dict], top_k: int = 5,
                  confidence_weighting: bool = True) -> dict:
        n = len(sensor_outputs)
        anomaly_flags = np.array([o["is_anomaly"] for o in sensor_outputs], dtype=float)
        drift_flags = np.array([o["drift_flag"] for o in sensor_outputs], dtype=float)
        retrain_flags = np.array([o["needs_retrain"] for o in sensor_outputs], dtype=float)
        confidences = np.array([o["confidence"] for o in sensor_outputs], dtype=float)

        anomaly_rate = float(anomaly_flags.mean()) if n else 0.0
        drift_rate = float(drift_flags.mean()) if n else 0.0
        retrain_rate = float(retrain_flags.mean()) if n else 0.0

        # top-K strength: mean confidence of the K most anomalous sensors (Eq. 14 input)
        order = np.argsort(-confidences)
        top_idx = order[:min(top_k, n)]
        top_k_strength = float(confidences[top_idx].mean()) if n else 0.0

        return {
            "anomaly_rate": anomaly_rate,
            "drift_rate": drift_rate,
            "retrain_rate": retrain_rate,
            "top_k_strength": top_k_strength,
            "system_anomaly": anomaly_rate > self.system_anomaly_threshold,
            "system_drift": drift_rate > self.system_drift_threshold,
            "system_needs_retrain": retrain_rate > self.system_needs_retrain_threshold,
        }
