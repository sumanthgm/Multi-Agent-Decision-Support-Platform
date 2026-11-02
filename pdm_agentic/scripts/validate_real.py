"""Check the method on a REAL labelled fault dataset (e.g. GPVS-Faults or the inverter-driven PMSM dataset).

The trained twin models predict system alarms, which real lab datasets do not label, so this script
re-trains the same pipeline (per-unit window features + XGBoost) on the real data with
leave-one-run-out testing, to show the representation works on real signals.

Input: a CSV with columns  timestamp, run_id, label, <signal columns...>
       --scale gives the rating used for per-unit, e.g.  --scale v_pv=500 i_pv=10
Example:
    python scripts/validate_real.py --csv data/real/gpvs_features.csv --scale vpv=300 ipv=10
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pdm.features import WINDOW, _rolling_slope  # noqa: E402


def window_feats(df: pd.DataFrame, cols, w: int, stride: int):
    out = []
    for c in cols:
        x = df[c].to_numpy(float)
        s = pd.Series(x).rolling(w, min_periods=w)
        out += [s.mean(), s.std(), s.min(), s.max(), pd.Series(_rolling_slope(x, w))]
    F = pd.concat(out, axis=1).to_numpy()[w - 1::stride]
    return F, df["label"].to_numpy()[w - 1::stride]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--scale", nargs="*", default=[], help="col=rating pairs for per-unit")
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--stride", type=int, default=5)
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    for kv in a.scale:
        k, v = kv.split("="); df[k] = df[k] / float(v)
    cols = [c for c in df.columns if c not in ("timestamp", "run_id", "label")]
    classes = sorted(df["label"].unique()); cmap = {c: i for i, c in enumerate(classes)}
    preds, trues = [], []
    for run in df["run_id"].unique():
        tr, te = df[df.run_id != run], df[df.run_id == run]
        Xtr, ytr = [], []
        for _, g in tr.groupby("run_id"):
            F, y = window_feats(g, cols, a.window, a.stride); Xtr.append(F); ytr.append(y)
        Xte, yte = window_feats(te, cols, a.window, a.stride)
        clf = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05)
        clf.fit(np.vstack(Xtr), np.vectorize(cmap.get)(np.concatenate(ytr)))
        preds.append(clf.predict(Xte)); trues.append(np.vectorize(cmap.get)(yte))
    p, t = np.concatenate(preds), np.concatenate(trues)
    print(classification_report(t, p, target_names=[str(c) for c in classes], zero_division=0))
    print("macro-F1 (leave-one-run-out):", round(f1_score(t, p, average="macro"), 3))
