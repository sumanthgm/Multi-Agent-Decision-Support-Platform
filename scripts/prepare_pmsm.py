"""Turn the inverter-driven PMSM fault dataset into the CSV layout scripts/validate_real.py reads.

Dataset: Zenodo 10.5281/zenodo.14482932 (CC-BY-4.0), "Comprehensive Dataset for Fault Detection and
Diagnosis in Inverter-Driven PMSM Systems". 10 Hz, 8 raw ADC signals (0-1023), 9 classes:
F0 normal, F1-F2 switch open-circuit, F3-F5 switch short-circuit, F6-F8 half-bridge overheating.
Each class is ONE continuous recording, so run_id = label.

    python scripts/prepare_pmsm.py                 # downloads the 1 MB zip into data/real/pmsm_raw/
    python scripts/prepare_pmsm.py --zip path/to/PMSM-inverter-fault-diagnosis-3.0.zip

Output: data/real/pmsm_10hz.csv with columns  timestamp, run_id, label, Ia, Ib, VDC, IDC, T1, T2, T3, VD.
Signals stay in ADC counts: the window features are min/max/mean/std/slope, and XGBoost splits are
unchanged by a fixed scale and offset per signal, so converting to amps or degC would not change results.
"""

import argparse
import zipfile
from pathlib import Path

import pandas as pd
import requests

URL = "https://zenodo.org/records/14482932/files/PMSM-inverter-fault-diagnosis-3.0.zip?download=1"
REAL = Path(__file__).resolve().parents[1] / "data" / "real"
MEMBER = "PMSM-inverter-fault-diagnosis-3.0/raw_data/dataset_CSV.csv"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=str(REAL / "pmsm_raw" / "pmsm.zip"))
    ap.add_argument("--out", default=str(REAL / "pmsm_10hz.csv"))
    a = ap.parse_args()
    zpath = Path(a.zip)
    if not zpath.exists():
        zpath.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(URL, timeout=120)
        r.raise_for_status()
        zpath.write_bytes(r.content)
    with zipfile.ZipFile(zpath) as z, z.open(MEMBER) as f:
        d = pd.read_csv(f).rename(columns={"FDD": "label"})
    seg = (d["label"] != d["label"].shift()).cumsum()
    if seg.nunique() != d["label"].nunique():
        raise SystemExit("expected one contiguous recording per class")
    d.insert(0, "run_id", d["label"])
    d.insert(0, "timestamp", d.groupby("run_id").cumcount() / 10.0)  # seconds from the start of the recording
    d.to_csv(a.out, index=False)
    print(f"wrote {len(d)} rows, {d['label'].nunique()} classes to {a.out}")
    print(d["label"].value_counts().sort_index().to_string())
