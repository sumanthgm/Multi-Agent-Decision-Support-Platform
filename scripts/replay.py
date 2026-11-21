"""Replay twin telemetry as a live feed.

    python scripts/replay.py --profile inverter_hybrid_10kw --mode api --speed 50    # post to the FastAPI service
    python scripts/replay.py --profile pump_drive_3hp --mode local --minutes 600      # run the platform in-process

The model trained on the 5 kW / 5 HP unit is reused for every rating through its profile.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pdm.twin import generate  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="inverter_hybrid_5kw")
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--mode", choices=["api", "local"], default="local")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--speed", type=float, default=100.0, help="rows per second (api mode)")
    ap.add_argument("--minutes", type=int, default=1440, help="rows to replay")
    a = ap.parse_args()
    d = generate(a.profile, a.days, a.seed, n_true=3, n_near=1)
    raw = d["raw"].iloc[: a.minutes]
    if a.mode == "api":
        import requests
        for ts, row in raw.iterrows():
            requests.post(f"{a.url}/ingest", params={"profile": a.profile},
                          json={"timestamp": str(ts), "values": row.to_dict()}, timeout=5)
            time.sleep(1.0 / a.speed)
    else:
        from pdm.agents.orchestrator import Platform
        from pdm.api_helpers import platform_for
        plat: Platform = platform_for(a.profile)
        last = None
        for t in range(60, len(raw), 5):
            out = plat.step(raw.iloc[max(0, t - 120):t])
            if out["state"] != "NORMAL" and out["state"] != last:
                print(json.dumps({k: out[k] for k in ("time", "state", "alarms", "warnings")}, default=str)[:400])
                if out["advice"]:
                    print("   advice:", out["advice"]["code"], "-", out["advice"]["steps"][0])
            last = out["state"]
