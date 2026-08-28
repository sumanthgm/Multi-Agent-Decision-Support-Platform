"""
Sensor Agent — paper Section II-B-1 (Eq. 2-5).

One instance per sensor. Each is a small univariate LSTM autoencoder (encoder-
decoder, latent_dim=4, time-distributed linear output) trained on normal windows
only. At runtime it scores reconstruction error, flags point anomalies via a
robust median+MAD threshold with cooldown, and flags distributional drift via
Jensen-Shannon divergence (KS-test fallback) with multi-vote confirmation.
"""
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp
import tensorflow as tf
from tensorflow.keras import layers, models


def build_lstm_autoencoder(window_len: int = 100, latent_dim: int = 4) -> tf.keras.Model:
    inp = layers.Input(shape=(window_len, 1))
    encoded = layers.LSTM(latent_dim)(inp)
    repeated = layers.RepeatVector(window_len)(encoded)
    decoded = layers.LSTM(latent_dim, return_sequences=True)(repeated)
    out = layers.TimeDistributed(layers.Dense(1))(decoded)
    model = models.Model(inp, out)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
    model.compile(optimizer=opt, loss="mse")
    return model


def train_sensor_autoencoder(X_normal_1d: np.ndarray, window_len: int = 100,
                              batch_size: int = 64, max_epochs: int = 50,
                              patience: int = 8) -> tf.keras.Model:
    """X_normal_1d: (N, window_len, 1) windows drawn ONLY from normal-labeled data."""
    model = build_lstm_autoencoder(window_len)
    es = tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)
    rlrop = tf.keras.callbacks.ReduceLROnPlateau(patience=max(2, patience // 2))
    model.fit(X_normal_1d, X_normal_1d, batch_size=batch_size, epochs=max_epochs,
              validation_split=0.1, callbacks=[es, rlrop], verbose=2)
    return model


class SensorAgent:
    """Runtime, streaming sensor agent. Call `.step(window)` once per incoming window."""

    def __init__(self, model: tf.keras.Model, cfg: dict, sensor_name: str = ""):
        self.model = model
        self.cfg = cfg
        self.name = sensor_name
        self.k = cfg.get("mad_sensitivity_k", 2.0)
        self.anomaly_cooldown = cfg.get("anomaly_cooldown_steps", 5)
        self.drift_jsd_threshold = cfg.get("drift_jsd_threshold", 0.10)
        self.drift_confirm_votes = cfg.get("drift_confirm_votes", 3)
        self.drift_cooldown_steps = cfg.get("drift_cooldown_steps", 100)
        self.drift_buffer_len = cfg.get("drift_buffer_len", 100)

        self.baseline_error_buffer = []   # populated from validation-set errors at init
        self.recent_error_buffer = []
        self._anomaly_cooldown_left = 0
        self._drift_cooldown_left = 0
        self._drift_vote_streak = 0
        self.median_e = None
        self.mad_e = None
        self._warmup_steps = 100
        self._steps_seen = 0

    def initialize_baseline(self, validation_errors: np.ndarray):
        self.baseline_error_buffer = list(validation_errors)
        self.median_e = float(np.median(validation_errors))
        self.mad_e = float(np.median(np.abs(validation_errors - self.median_e))) + 1e-8

    def _reconstruction_error(self, window_1d: np.ndarray) -> float:
        x = window_1d.reshape(1, -1, 1)
        recon = self.model.predict(x, verbose=0)[0, :, 0]
        return float(np.mean((window_1d - recon) ** 2))  # Eq. (per Sec II-B-1)

    def _confidence(self, e_t: float) -> float:
        z = (e_t - (self.median_e + self.k * self.mad_e)) / self.mad_e  # Eq. 5
        return float(1 / (1 + np.exp(-z)))  # Eq. 4, sigmoid

    def _check_drift(self) -> bool:
        if len(self.recent_error_buffer) < self.drift_buffer_len:
            return False
        baseline = np.array(self.baseline_error_buffer[-self.drift_buffer_len:]
                             or self.baseline_error_buffer)
        recent = np.array(self.recent_error_buffer[-self.drift_buffer_len:])
        try:
            bins = np.histogram_bin_edges(np.concatenate([baseline, recent]), bins=20)
            p, _ = np.histogram(baseline, bins=bins, density=True)
            q, _ = np.histogram(recent, bins=bins, density=True)
            p = p / (p.sum() + 1e-12)
            q = q / (q.sum() + 1e-12)
            jsd = jensenshannon(p + 1e-12, q + 1e-12) ** 2
            drifted = jsd > self.drift_jsd_threshold
        except Exception:
            _, p_val = ks_2samp(baseline, recent)  # fallback
            drifted = p_val < 0.05
        return bool(drifted)

    def step(self, window_1d: np.ndarray, step_idx: int) -> dict:
        self._steps_seen += 1
        e_t = self._reconstruction_error(window_1d)

        # rolling robust stats: fixed during warmup, updated every 10 steps after
        if self._steps_seen > self._warmup_steps and self._steps_seen % 10 == 0:
            recent50 = (self.recent_error_buffer[-50:] or [e_t])
            self.median_e = float(np.median(recent50))
            self.mad_e = float(np.median(np.abs(np.array(recent50) - self.median_e))) + 1e-8

        is_anomaly = False
        if self._anomaly_cooldown_left == 0:
            if e_t > self.median_e + self.k * self.mad_e:  # Eq. 3
                is_anomaly = True
                self._anomaly_cooldown_left = self.anomaly_cooldown
        else:
            self._anomaly_cooldown_left -= 1

        self.recent_error_buffer.append(e_t)
        if len(self.recent_error_buffer) > 500:
            self.recent_error_buffer = self.recent_error_buffer[-500:]

        drift_flag = False
        if self._drift_cooldown_left == 0:
            if self._check_drift():
                self._drift_vote_streak += 1
            else:
                self._drift_vote_streak = 0
            if self._drift_vote_streak >= self.drift_confirm_votes:
                drift_flag = True
                self._drift_cooldown_left = self.drift_cooldown_steps
                self._drift_vote_streak = 0
        else:
            self._drift_cooldown_left -= 1

        confidence = self._confidence(e_t)

        recent_anom_rate = float(np.mean(
            [1 if e > self.median_e + self.k * self.mad_e else 0
             for e in self.recent_error_buffer[-50:]]
        )) if self.recent_error_buffer else 0.0
        needs_retrain = (
            (recent_anom_rate > self.cfg.get("retrain_recent_anomaly_rate", 0.30))
            + drift_flag
            + (e_t > self.cfg.get("retrain_error_multiplier", 2.0) * self.median_e)
        ) >= 2  # "at least two of the following hold"

        return {
            "sensor": self.name,
            "reconstruction_error": e_t,
            "is_anomaly": is_anomaly,
            "drift_flag": drift_flag,
            "confidence": confidence,
            "needs_retrain": bool(needs_retrain),
        }
