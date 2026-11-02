import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fastapi.testclient import TestClient  # noqa: E402

from pdm.serve.api import app  # noqa: E402
from pdm.twin import generate  # noqa: E402

client = TestClient(app)


def test_profiles_carry_raw_limits():
    profs = {p["name"]: p for p in client.get("/profiles").json()}
    assert len(profs) == 6
    inv = {a["code"]: a for a in profs["inverter_hybrid_5kw"]["alarms"]}
    assert inv["INV-11"]["raw_column"] == "t_hs" and inv["INV-11"]["raw_limits"] == [85.0]
    assert inv["INV-03"]["raw_limits"] == [49.5, 50.5]  # frequency band around 50 Hz


def test_assets_are_kept_apart():
    raw = generate("pump_drive_3hp", 1, 3, n_true=0, n_near=0)["raw"].iloc[480:560]
    for ts, row in raw.iterrows():
        client.post(
            "/ingest", params={"profile": "pump_drive_3hp"}, json={"timestamp": str(ts), "values": row.to_dict()}
        )
    s = client.get("/status", params={"profile": "pump_drive_3hp"}).json()
    assert s["state"] in {"NORMAL", "WARNING", "ALARM"} and s["rows"] == 80
    assert client.get("/status", params={"profile": "pump_drive_10hp"}).json()["status"] == "no data"
    f = client.get("/forecast", params={"profile": "pump_drive_3hp"}).json()
    assert len(f["alarms"]) == 8 and len(f["alarms"][0]["forecast_pu"]) == f["horizon"]
    assert client.get("/status", params={"profile": "nope"}).status_code == 404


def test_feed_and_advice():
    feed = client.post("/feed/start", params={"profile": "inverter_hybrid_3kw", "speed": 50, "seed": 1}).json()
    assert feed["running"]
    assert len(client.get("/history", params={"profile": "inverter_hybrid_3kw", "n": 1000}).json()) >= 240
    assert not client.post("/feed/stop", params={"profile": "inverter_hybrid_3kw"}).json()["running"]
    adv = client.get("/advice", params={"profile": "inverter_hybrid_3kw", "code": "INV-11"}).json()
    assert adv["steps"] and "llm_summary" not in adv and "inverter_hybrid_3kw" in adv["context"]
