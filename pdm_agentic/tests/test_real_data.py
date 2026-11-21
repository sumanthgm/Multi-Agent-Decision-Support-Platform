import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pdm.twin import generate, load_weather, weather_block  # noqa: E402


def _weather_csv(tmp_path: Path) -> Path:
    idx = pd.date_range("2022-01-01", periods=4 * 1440, freq="1min")
    w = pd.DataFrame({"g": np.clip(np.sin(np.pi * ((idx.hour + idx.minute / 60) - 6) / 12), 0, None) * 900,
                      "t_amb": 25.0}, index=idx)
    w = w.drop(idx[100:110])                              # short gap on day 1: interpolated
    w = w.drop(idx[1440 + 600:1440 + 700])                # 100-min gap on day 2: day dropped
    w.index.name = "time"
    path = tmp_path / "w.csv"
    w.to_csv(path)
    return path


def test_measured_weather_keeps_complete_days(tmp_path):
    w = load_weather(_weather_csv(tmp_path))
    days = w.index.normalize().unique()
    assert [str(d.date()) for d in days] == ["2022-01-01", "2022-01-03", "2022-01-04"]
    assert len(w) == 3 * 1440 and not w.isna().any().any()
    b = weather_block(w, 1, 2)
    assert len(b) == 2 * 1440 and b.index.is_monotonic_increasing and b.index.freq == "min"
    d = generate("pump_drive_5hp", 2, 3, n_true=0, n_near=0, weather=w.iloc[:2 * 1440])
    assert len(d["raw"]) == 2 * 1440


def test_blocked_split_never_shares_samples():
    spec = importlib.util.spec_from_file_location("validate_real", ROOT / "scripts" / "validate_real.py")
    vr = importlib.util.module_from_spec(spec); spec.loader.exec_module(vr)
    df = pd.DataFrame({"run_id": ["a"] * 100 + ["b"] * 37})
    w = 10
    pos = df.groupby("run_id").cumcount().to_numpy()
    for tr, te in vr.make_folds(df, "blocks", 4, w):
        assert te.any() and not (tr & te).any()
        for run in ("a", "b"):
            r = (df["run_id"] == run).to_numpy()
            test_rows = {p for e in pos[te & r] for p in range(e - w + 1, e + 1)}
            train_rows = {p for e in pos[tr & r] for p in range(e - w + 1, e + 1)}
            assert not test_rows & train_rows
