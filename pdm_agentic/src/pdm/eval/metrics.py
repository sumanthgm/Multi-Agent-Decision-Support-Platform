"""Window-level and event-level metrics for early warning."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from ..features import STRIDE


def evaluate(codes: list[str], y: np.ndarray, m: np.ndarray, pred: np.ndarray, prob: np.ndarray,
             labels: pd.DataFrame, ends: np.ndarray, horizons: dict[str, int]) -> pd.DataFrame:
    """Per-alarm metrics. Events are alarm onsets; lead time is minutes from first warning to onset."""
    rows = []
    n_days = len(labels) / 1440.0
    for j, code in enumerate(codes):
        ok = m[:, j] > 0
        yt, yp, pp = y[ok, j], pred[ok, j], prob[ok, j]
        r = dict(alarm=code)
        r["precision"] = precision_score(yt, yp, zero_division=0)
        r["recall"] = recall_score(yt, yp, zero_division=0)
        r["f1"] = f1_score(yt, yp, zero_division=0)
        r["pr_auc"] = average_precision_score(yt, pp) if yt.sum() > 0 else np.nan

        onset_idx = np.flatnonzero(labels[f"{code}_onset"].to_numpy())
        H = horizons[code]
        detected, leads = 0, []
        fired = pred[:, j] > 0
        for t_o in onset_idx:
            win = (ends >= t_o - H) & (ends < t_o)
            hits = np.flatnonzero(win & fired)
            if len(hits):
                detected += 1
                leads.append(t_o - ends[hits[0]])
        r["events"] = len(onset_idx)
        r["event_recall"] = detected / len(onset_idx) if len(onset_idx) else np.nan
        r["median_lead_min"] = float(np.median(leads)) if leads else np.nan

        # alerts = runs of consecutive firing windows; false if no onset within H after alert start
        starts = np.flatnonzero(fired & ~np.concatenate([[False], fired[:-1]]))
        false = 0
        for s in starts:
            t_s = ends[s]
            if not ((onset_idx > t_s) & (onset_idx <= t_s + H + STRIDE)).any():
                false += 1
        r["false_alerts_per_day"] = false / n_days
        rows.append(r)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    return {
        "macro_f1": df["f1"].mean(), "macro_pr_auc": df["pr_auc"].mean(),
        "event_recall": df["event_recall"].mean(), "median_lead_min": df["median_lead_min"].median(),
        "false_alerts_per_day": df["false_alerts_per_day"].sum(),
    }
