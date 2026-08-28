"""
Adaptive Window Agent — paper Section II-B-3 (Eq. 6-12).

Two frozen sub-models:
  1. GDRNet-style MLP regressor: predicts the optimal sliding window size W_bar_t
     from flattened window-level feature vectors. (Paper cites its own prior work,
     "GDRNet", IEEE Access 2025, https://ieeexplore.ieee.org/document/11259050 —
     full architecture there; only the MLP width/activation/training recipe is
     given in THIS paper and is reproduced below.)
  2. LSTM next-step predictor (M_NSP): forecasts the next multivariate timestep,
     used to compute forecast error e_t and the Forecast Deviation Score (FDS).

From these it derives, every step:
  - W_bar_t   : predicted optimal window length
  - FDS_t     : Forecast Deviation Score (Eq. 9)
  - FDI_t     : Forecast Drift Index, JSD over FDS distributions (Eq. 10)
  - WDI_t     : Window Drift Index, JSD over predicted-window-size distributions (Eq. 11)
  - WSS_t     : Window Shift Score (Eq. 12)
  - event_t   : {ANOMALY, DRIFT, stable}
"""
import numpy as np
from scipy.spatial.distance import jensenshannon
import tensorflow as tf
from tensorflow.keras import layers, models


def build_gdrnet_mlp(input_dim: int) -> tf.keras.Model:
    inp = layers.Input(shape=(input_dim,))
    x = inp
    widths = [64, 32, 16, 8]
    for i, w in enumerate(widths):
        x = layers.Dense(w)(x)
        x = layers.LeakyReLU(negative_slope=0.1)(x)
        if i == 0:
            x = layers.Dropout(0.1)(x)
    out = layers.Dense(1, activation="linear")(x)
    model = models.Model(inp, out)
    opt = tf.keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0)
    model.compile(optimizer=opt, loss=tf.keras.losses.Huber())
    return model


def build_next_step_predictor(n_sensors: int) -> tf.keras.Model:
    inp = layers.Input(shape=(None, n_sensors))  # variable-length input
    x = layers.LSTM(128)(inp)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(n_sensors, activation="linear")(x)
    model = models.Model(inp, out)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=opt, loss="mse")
    return model


class AdaptiveWindowAgent:
    def __init__(self, gdrnet_model, nsp_model, cfg: dict):
        self.gdrnet = gdrnet_model
        self.nsp = nsp_model
        self.k = cfg.get("anomaly_k", 3.0)
        self.anomaly_cooldown_steps = cfg.get("anomaly_cooldown_steps", 5)
        self.fdi_bins = cfg.get("fdi_hist_bins", 25)
        self.wdi_bins = cfg.get("wdi_hist_bins", 20)
        self.fdi_threshold = cfg.get("fdi_threshold", 0.25)
        self.wdi_threshold = cfg.get("wdi_threshold", 0.20)
        self.drift_confirm_votes = cfg.get("drift_confirm_votes", 10)
        self.drift_cooldown_steps = cfg.get("drift_cooldown_steps", 100)
        self.long_term_len = cfg.get("long_term_buffer", 300)
        self.recent_len = cfg.get("recent_buffer", 50)
        self.min_window_clamp = cfg.get("min_window_clamp", 2)

        # baseline stats — populate via `fit_baseline()` from training-set errors
        self.median_e, self.mad_e = 0.0, 1.0
        self.mu_w, self.sigma_w = 50.0, 10.0
        self.fds_long_term, self.window_long_term = [], []
        self.fds_recent, self.window_recent = [], []
        self._anomaly_cooldown_left = 0
        self._drift_cooldown_left = 0
        self._drift_vote_streak = 0

    def fit_baseline(self, baseline_forecast_errors: np.ndarray,
                      baseline_window_sizes: np.ndarray):
        self.median_e = float(np.median(baseline_forecast_errors))
        self.mad_e = float(np.median(np.abs(baseline_forecast_errors - self.median_e))) + 1e-8
        self.mu_w = float(np.mean(baseline_window_sizes))
        self.sigma_w = float(np.std(baseline_window_sizes)) + 1e-8
        self.fds_long_term = list(
            (baseline_forecast_errors - self.median_e) / self.mad_e
        )[-self.long_term_len:]
        self.window_long_term = list(baseline_window_sizes)[-self.long_term_len:]

    @staticmethod
    def _jsd_over_samples(a, b, bins):
        if len(a) < 2 or len(b) < 2:
            return 0.0
        edges = np.histogram_bin_edges(np.concatenate([a, b]), bins=bins)
        p, _ = np.histogram(a, bins=edges, density=True)
        q, _ = np.histogram(b, bins=edges, density=True)
        p = p / (p.sum() + 1e-12)
        q = q / (q.sum() + 1e-12)
        return float(jensenshannon(p + 1e-12, q + 1e-12) ** 2)

    def step(self, window_features: np.ndarray, window_multivariate: np.ndarray) -> dict:
        """
        window_features: flattened feature vector for the GDRNet MLP input.
        window_multivariate: (T, S) recent multivariate sequence for the NSP model.
        """
        w_bar = float(self.gdrnet.predict(window_features[None, :], verbose=0)[0, 0])
        w_bar = max(w_bar, self.min_window_clamp)

        w_len = max(int(round(w_bar)), 1)
        seq_in = window_multivariate[-w_len:][None, ...]
        y_pred = self.nsp.predict(seq_in, verbose=0)[0]
        y_true = window_multivariate[-1]
        e_t = float(np.mean((y_pred - y_true) ** 2))  # Eq. 8, MSE

        fds_t = (e_t - self.median_e) / self.mad_e  # Eq. 9
        wss_t = (w_bar - self.mu_w) / self.sigma_w   # Eq. 12

        self.fds_recent.append(fds_t)
        self.window_recent.append(w_bar)
        self.fds_recent = self.fds_recent[-self.recent_len:]
        self.window_recent = self.window_recent[-self.recent_len:]

        fdi_t = self._jsd_over_samples(np.array(self.fds_long_term),
                                        np.array(self.fds_recent), self.fdi_bins)  # Eq. 10
        wdi_t = self._jsd_over_samples(np.array(self.window_long_term),
                                        np.array(self.window_recent), self.wdi_bins)  # Eq. 11

        is_anomaly = False
        if self._anomaly_cooldown_left == 0:
            if abs(fds_t) > self.k:
                is_anomaly = True
                self._anomaly_cooldown_left = self.anomaly_cooldown_steps
        else:
            self._anomaly_cooldown_left -= 1

        event = "stable"
        if self._drift_cooldown_left == 0:
            drift_vote = (wdi_t > self.wdi_threshold) or (fdi_t > self.fdi_threshold)
            self._drift_vote_streak = self._drift_vote_streak + 1 if drift_vote else 0
            if self._drift_vote_streak >= self.drift_confirm_votes:
                event = "DRIFT"
                self._drift_cooldown_left = self.drift_cooldown_steps
                self._drift_vote_streak = 0
            elif is_anomaly:
                event = "ANOMALY"
        else:
            self._drift_cooldown_left -= 1

        # WDI is treated as the dominant drift signal (paper, Sec II-B-3)
        return {
            "w_bar": w_bar, "forecast_error": e_t, "FDS": fds_t, "FDI": fdi_t,
            "WSS": wss_t, "WDI": wdi_t, "event": event,
        }
