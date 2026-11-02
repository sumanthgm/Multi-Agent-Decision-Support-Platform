"""Early-Warning Agent: one gradient-boosted classifier per alarm on window features.

Predicts, every 5 minutes, the probability that each alarm starts within its warning horizon.
Decision thresholds are tuned per alarm on a time-ordered validation split (last 20% of training).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from xgboost import XGBClassifier


def tune_threshold(y: np.ndarray, p: np.ndarray) -> float:
    if y.sum() == 0:
        return 0.5
    grid = np.linspace(0.05, 0.95, 37)
    scores = [f1_score(y, p >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


class EarlyWarningXGB:
    def __init__(self, codes: list[str], seed: int = 0, **xgb_kw):
        self.codes = codes
        self.seed = seed
        self.kw = dict(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.9,
                       colsample_bytree=0.8, tree_method="hist", n_jobs=2, eval_metric="logloss")
        self.kw.update(xgb_kw)
        self.models: dict[str, XGBClassifier | None] = {}
        self.thresholds: dict[str, float] = {}
        self.columns: list[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray, m: np.ndarray, val_frac: float = 0.2) -> "EarlyWarningXGB":
        self.columns = list(X.columns)
        n = len(X)
        cut = int(n * (1 - val_frac))
        Xv = X.to_numpy(dtype=np.float32)
        for j, code in enumerate(self.codes):
            ok = m[:, j] > 0
            tr = ok.copy(); tr[cut:] = False
            va = ok.copy(); va[:cut] = False
            yt = y[tr, j]
            if yt.sum() < 3:
                self.models[code] = None; self.thresholds[code] = 0.5
                continue
            spw = float((yt == 0).sum() / max(yt.sum(), 1))
            clf = XGBClassifier(scale_pos_weight=spw, random_state=self.seed, **self.kw)
            clf.fit(Xv[tr], yt)
            pv = clf.predict_proba(Xv[va])[:, 1] if va.any() else np.array([])
            self.thresholds[code] = tune_threshold(y[va, j], pv) if len(pv) else 0.5
            clf.fit(Xv[ok], y[ok, j])                     # refit on all valid training windows
            self.models[code] = clf
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xv = X[self.columns].to_numpy(dtype=np.float32)
        out = np.zeros((len(X), len(self.codes)), dtype=np.float32)
        for j, code in enumerate(self.codes):
            clf = self.models.get(code)
            if clf is not None:
                out[:, j] = clf.predict_proba(Xv)[:, 1]
        return out

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        p = self.predict_proba(X)
        th = np.array([self.thresholds[c] for c in self.codes])
        return (p >= th[None, :]).astype(int)
