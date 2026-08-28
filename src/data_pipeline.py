"""
MetroPT-3 data pipeline — paper Sections III-B (Dataset) and III-C (Preprocessing
and Label Construction).

Steps implemented exactly as described in the paper:
  1. Load raw MetroPT-3 (15 sensor channels, 1,516,948 obs, 10s sampling).
  2. Sensor selection via Granger-causality correlation analysis -> drop
     Pressure_switch, Caudal_impulses, Oil_level (12 sensors retained).
  3. Exclude data before 2020-04-01 (unreliable/constant early period).
  4. Impute "?" -> NaN -> one-day seasonal lag.
  5. Per-feature Min-Max scaling to [-1, 1], parameters stored per feature.
  6. Sliding windows, W=100, stride=1, window labeled by its final timestamp.
  7. Point-level labels: fault-in-progress (within failure interval) and
     early-warning (within H=2h preceding a failure). Window label = any-point
     rule over {Normal, Warning, Fault}, Fault overrides Warning.
  8. Chronology-aware, fault-aware split: train ends before Failure 3 early-warning
     start; test = Failure 3 interval; holdout = everything after (covers Failure 4).

NOTE: exact global window/row counts are not published beyond the aggregate
1,516,948-observation total and the four failure windows (Table 2), so your
train/test/holdout window counts will not match the paper's internal numbers
bit-for-bit — only the *split logic* is specified precisely enough to reproduce.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/metropt3_raw.csv")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SENSORS_KEPT = [
    "TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature",
    "Motor_current", "COMP", "DV_electric", "Towers", "MPG", "LPS",
]
SENSORS_DROPPED = ["Pressure_switch", "Caudal_impulses", "Oil_level"]

# Table 2 — failure intervals as provided by MetroPT-3 (paper transcription)
FAILURES = [
    {"id": 1, "start": "2020-04-18 00:00", "end": "2020-04-18 23:59"},
    {"id": 2, "start": "2020-05-29 23:30", "end": "2020-05-30 06:00"},
    {"id": 3, "start": "2020-06-05 10:00", "end": "2020-06-07 14:30"},
    {"id": 4, "start": "2020-07-15 14:30", "end": "2020-07-15 19:00"},
]
EARLY_WARNING_HORIZON = pd.Timedelta(hours=2)
EXCLUDE_BEFORE = pd.Timestamp("2020-04-01")
WINDOW_LEN = 100
STRIDE = 1


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(
            "Expected a 'timestamp' column in the raw CSV. Rename your UCI export's "
            "time column to 'timestamp' before running this pipeline."
        )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def select_sensors_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = ["timestamp"] + [c for c in SENSORS_KEPT if c in df.columns]
    missing = set(SENSORS_KEPT) - set(df.columns)
    if missing:
        print(f"[warn] expected sensor columns not found in raw CSV: {missing}")
    df = df[keep_cols].copy()
    df = df[df["timestamp"] >= EXCLUDE_BEFORE].reset_index(drop=True)
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace("?", np.nan)
    for c in SENSORS_KEPT:
        if c not in df.columns:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # one-day seasonal lag imputation: fill NaN with the value from exactly
    # one day (assumes ~10s sampling => 8640 rows/day; adjust if your export differs)
    ROWS_PER_DAY = 8640
    for c in SENSORS_KEPT:
        if c not in df.columns:
            continue
        lag = df[c].shift(ROWS_PER_DAY)
        df[c] = df[c].fillna(lag)
    df = df.ffill().bfill()  # residual edge NaNs (paper doesn't specify; documented fallback)
    return df


def minmax_scale(df: pd.DataFrame):
    scale_params = {}
    df_scaled = df.copy()
    for c in SENSORS_KEPT:
        if c not in df.columns:
            continue
        lo, hi = df[c].min(), df[c].max()
        scale_params[c] = {"min": float(lo), "max": float(hi)}
        rng = (hi - lo) if hi != lo else 1.0
        df_scaled[c] = 2 * (df[c] - lo) / rng - 1
    return df_scaled, scale_params


def label_points(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["point_label"] = 0  # 0 Normal, 1 Warning, 2 Fault
    for f in FAILURES:
        start, end = pd.Timestamp(f["start"]), pd.Timestamp(f["end"])
        warn_start = start - EARLY_WARNING_HORIZON
        in_fault = (df["timestamp"] >= start) & (df["timestamp"] <= end)
        in_warning = (df["timestamp"] >= warn_start) & (df["timestamp"] < start)
        df.loc[in_warning, "point_label"] = np.maximum(df.loc[in_warning, "point_label"], 1)
        df.loc[in_fault, "point_label"] = 2  # Fault overrides Warning
    return df


def make_windows(df: pd.DataFrame, feature_cols):
    """Any-point rule: window label = max class label among points in window."""
    values = df[feature_cols].values
    labels = df["point_label"].values
    timestamps = df["timestamp"].values
    n = len(df)
    X, y, end_ts = [], [], []
    for end in range(WINDOW_LEN - 1, n, STRIDE):
        start = end - WINDOW_LEN + 1
        X.append(values[start:end + 1])
        y.append(labels[start:end + 1].max())
        end_ts.append(timestamps[end])
    return np.array(X), np.array(y), np.array(end_ts)


def chronology_aware_split(end_ts, y):
    """Train precedes Failure 3 warning start; test spans Failure 3; holdout = rest."""
    f3 = FAILURES[2]
    f3_warn_start = pd.Timestamp(f3["start"]) - EARLY_WARNING_HORIZON
    f3_end = pd.Timestamp(f3["end"])
    end_ts = pd.to_datetime(end_ts)
    train_mask = end_ts < f3_warn_start
    test_mask = (end_ts >= f3_warn_start) & (end_ts <= f3_end)
    holdout_mask = end_ts > f3_end
    return train_mask, test_mask, holdout_mask


def run():
    print("1/6 loading raw data...")
    df = load_raw()
    print("2/6 selecting sensors + date filter...")
    df = select_sensors_and_filter(df)
    print("3/6 imputing missing values...")
    df = impute_missing(df)
    print("4/6 min-max scaling...")
    df, scale_params = minmax_scale(df)
    print("5/6 labeling points + windowing...")
    df = label_points(df)
    feature_cols = [c for c in SENSORS_KEPT if c in df.columns]
    X, y, end_ts = make_windows(df, feature_cols)
    print(f"    windows: {X.shape}, label distribution: {np.bincount(y)}")
    print("6/6 chronology-aware split...")
    train_mask, test_mask, holdout_mask = chronology_aware_split(end_ts, y)

    np.savez_compressed(
        OUT_DIR / "windows.npz",
        X_train=X[train_mask], y_train=y[train_mask],
        X_test=X[test_mask], y_test=y[test_mask],
        X_holdout=X[holdout_mask], y_holdout=y[holdout_mask],
        feature_cols=np.array(feature_cols),
    )
    import json
    with open(OUT_DIR / "scale_params.json", "w") as f:
        json.dump(scale_params, f, indent=2)
    print(f"Saved to {OUT_DIR}/windows.npz and scale_params.json")
    print(f"train={train_mask.sum()} test={test_mask.sum()} holdout={holdout_mask.sum()}")


if __name__ == "__main__":
    run()
