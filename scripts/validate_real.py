"""Check the method on a REAL labelled fault dataset (inverter-driven PMSM, GPVS-Faults, ...).

The twin models predict system alarms, which real lab datasets do not label, so this script re-trains
the same pipeline (rolling window features: mean, std, min, max, slope + XGBoost) on the real signals.

Input: a CSV with columns  timestamp, run_id, label, <signal columns...>   (see scripts/prepare_pmsm.py)

Splits:
  --split run     leave-one-run-out; needs several runs per class (e.g. GPVS-Faults).
  --split blocks  each run is cut into --folds contiguous time blocks; fold k tests block k of every run
                  and trains on windows lying fully outside it (no shared samples). Use this when every
                  class is a single recording, as in the PMSM dataset.
Baseline: --features instant uses only the latest sample, to show what the window adds.

Examples:
    python scripts/validate_real.py --csv data/real/pmsm_10hz.csv --split blocks --window 20 --out results/real_faults
    python scripts/validate_real.py --csv data/real/pmsm_10hz.csv --split blocks --window 20 --features instant
    python scripts/validate_real.py --csv data/real/pmsm_10hz.csv --split blocks --window 20 \
        --signals Ia Ib VDC IDC VD --labels F0 F1 F2 F3 F4 F5 --tag electrical
    python scripts/validate_real.py --csv data/real/gpvs_features.csv --split run --scale vpv=300 ipv=10
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pdm.features import WINDOW, _rolling_slope  # noqa: E402


def run_features(g: pd.DataFrame, cols, w: int, kind: str) -> np.ndarray:
    """One row per sample: features of the window ending there (NaN until w samples are in)."""
    if kind == "instant":
        return g[cols].to_numpy(float)
    out = []
    for c in cols:
        x = g[c].to_numpy(float)
        s = pd.Series(x).rolling(w, min_periods=w)
        out += [s.mean(), s.std(), s.min(), s.max(), pd.Series(_rolling_slope(x, w))]
    return pd.concat(out, axis=1).to_numpy()


def make_folds(df: pd.DataFrame, split: str, folds: int, w: int):
    """Yield (train_mask, test_mask) over window ends; windows never straddle the train/test boundary."""
    pos = df.groupby("run_id").cumcount().to_numpy()
    n = df.groupby("run_id")["run_id"].transform("size").to_numpy()
    if split == "run":
        for run in df["run_id"].unique():
            te = (df["run_id"] == run).to_numpy()
            yield ~te, te
        return
    for k in range(folds):
        lo, hi = n * k // folds, n * (k + 1) // folds          # test block [lo, hi) inside each run
        te = (pos >= lo + w - 1) & (pos < hi)                   # window fully inside the block
        tr = (pos < lo) | (pos - w + 1 >= hi)                   # window fully outside it
        yield tr, te


def classify(Xtr, ytr, Xte, n_classes):
    clf = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, n_jobs=2,
                        objective="multi:softprob" if n_classes > 2 else "binary:logistic")
    clf.fit(Xtr, ytr)
    return clf.predict(Xte)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--scale", nargs="*", default=[], help="col=rating pairs for per-unit")
    ap.add_argument("--split", choices=["run", "blocks"], default="run")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--window", type=int, default=WINDOW, help="samples per window")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--features", choices=["window", "instant"], default="window")
    ap.add_argument("--signals", nargs="*", help="signal columns to use (default: all)")
    ap.add_argument("--labels", nargs="*", help="keep only these classes")
    ap.add_argument("--normal", default="F0", help="label of healthy operation, for the detection score")
    ap.add_argument("--out", help="folder for a JSON + markdown summary")
    ap.add_argument("--tag", default="", help="name for this run in the --out files")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    df["label"] = df["label"].astype(str)
    if a.labels:
        df = df[df["label"].isin(a.labels)]
    df = df.sort_values(["run_id", "timestamp"], kind="stable").reset_index(drop=True)
    for kv in a.scale:
        k, v = kv.split("=")
        df[k] = df[k] / float(v)
    cols = a.signals or [c for c in df.columns if c not in ("timestamp", "run_id", "label")]
    classes = sorted(df["label"].unique())
    y_all = df["label"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    w = 1 if a.features == "instant" else a.window
    X_all = np.vstack([run_features(g, cols, w, a.features) for _, g in df.groupby("run_id", sort=False)])
    every = (df.groupby("run_id").cumcount().to_numpy() % a.stride) == 0

    preds, trues = [], []
    for tr, te in make_folds(df, a.split, a.folds, w):
        tr, te = tr & every, te & every
        if not te.any() or len(np.unique(y_all[tr])) < len(classes):
            print("skipping a fold: some class has no training windows"); continue
        preds.append(classify(X_all[tr], y_all[tr], X_all[te], len(classes))); trues.append(y_all[te])
    p, t = np.concatenate(preds), np.concatenate(trues)

    names = [str(c) for c in classes]
    print(classification_report(t, p, labels=range(len(names)), target_names=names, zero_division=0))
    macro = f1_score(t, p, average="macro")
    res = {"tag": a.tag or a.features, "csv": a.csv, "split": a.split, "folds": a.folds if a.split == "blocks" else None,
           "window": w, "features": a.features, "signals": cols, "classes": names, "test_windows": int(len(t)),
           "macro_f1": round(float(macro), 3), "accuracy": round(float((p == t).mean()), 3)}
    if a.normal in names:
        k = names.index(a.normal)
        res["detection_f1"] = round(float(f1_score(t != k, p != k)), 3)          # any fault vs normal
    print(f"macro-F1 ({a.split} split): {res['macro_f1']}   accuracy: {res['accuracy']}"
          + (f"   fault detection F1: {res['detection_f1']}" if "detection_f1" in res else ""))

    if a.out:
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        stem = f"real_{Path(a.csv).stem}_{res['tag']}"
        (out / f"{stem}.json").write_text(json.dumps(res, indent=2))
        cm = pd.DataFrame(confusion_matrix(t, p, labels=range(len(names))), index=names, columns=names)
        cm.to_csv(out / f"{stem}_confusion.csv")
        print(f"saved {out / stem}.json and _confusion.csv")
