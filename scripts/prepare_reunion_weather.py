"""Reduce the La Reunion rooftop-PV weather file (1 s, ~2.2 GB) to the twin's 1-minute weather CSV.

Dataset: Zenodo 10.5281/zenodo.7157424 (CC-BY-4.0), file dt1_solar_and_meteorological_measurement.csv
         columns  time, DTI, GTI, TA, TPV   (GTI = plane-of-array irradiance W/m2, TA = ambient degC,
         timestamps in local time, UTC+4)

    python scripts/prepare_reunion_weather.py                      # stream from Zenodo (no 2.2 GB file kept)
    python scripts/prepare_reunion_weather.py --src path/to/dt1_solar_and_meteorological_measurement.csv

Output: data/real/reunion_weather_1min.csv with columns  time, g, t_amb  (1-minute means, local time).
Load it with pdm.twin.load_weather(), which fills short gaps and keeps complete days only.
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import requests

URL = "https://zenodo.org/records/7828879/files/dt1_solar_and_meteorological_measurement.csv?download=1"
OUT = Path(__file__).resolve().parents[1] / "data" / "real" / "reunion_weather_1min.csv"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=URL, help="local CSV path or URL (default: Zenodo)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--chunk", type=int, default=2_000_000, help="rows per chunk")
    a = ap.parse_args()

    if a.src.startswith("http"):
        resp = requests.get(a.src, stream=True, timeout=60)
        resp.raise_for_status()
        resp.raw.decode_content = True
        src = resp.raw
    else:
        src = a.src
    parts, rows, t0 = [], 0, time.time()
    for chunk in pd.read_csv(src, usecols=["time", "GTI", "TA"], chunksize=a.chunk):
        minute = pd.to_datetime(chunk["time"].str[:16], format="%Y-%m-%d %H:%M")  # drop seconds and +04:00
        g = chunk.groupby(minute)[["GTI", "TA"]]
        parts.append(pd.concat([g.sum(), g.count().add_suffix("_n")], axis=1))
        rows += len(chunk)
        print(f"{rows / 1e6:6.1f} M rows  up to {minute.iloc[-1]}  ({time.time() - t0:.0f} s)", flush=True)
    s = pd.concat(parts).groupby(level=0).sum()          # minutes split across chunks are merged here
    out = pd.DataFrame({"g": (s["GTI"] / s["GTI_n"]).clip(lower=0), "t_amb": s["TA"] / s["TA_n"]})
    out.index.name = "time"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.round(2).to_csv(a.out)
    print(f"wrote {len(out)} minutes ({out.index[0]} .. {out.index[-1]}) to {a.out}")
