import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pdm.agents.advisor import AdvisorAgent  # noqa: E402
from pdm.agents.orchestrator import Platform  # noqa: E402
from pdm.features import pu_features, targets, window_ends  # noqa: E402
from pdm.agents.early_warning import EarlyWarningXGB  # noqa: E402
from pdm.profiles import Profile, list_profiles  # noqa: E402
from pdm.twin import generate  # noqa: E402


def test_profiles_load():
    names = list_profiles()
    assert len(names) == 6
    for n in names:
        p = Profile.load(n)
        assert len(p.alarms) >= 8


def test_twin_has_labelled_events():
    d = generate("inverter_hybrid_5kw", days=10, seed=5, n_true=3, n_near=1)
    onsets = sum(int(d["labels"][f"{c}_onset"].sum()) for c in d["profile"].alarm_codes)
    assert onsets > 10
    assert set(d["labels"]["state"].unique()) <= {0, 1, 2}


def test_per_unit_is_rating_independent():
    a = generate("inverter_hybrid_3kw", 2, 7, n_true=0, n_near=0)["pu"]
    b = generate("inverter_hybrid_10kw", 2, 7, n_true=0, n_near=0)["pu"]
    # same weather and seed: load in pu has the same distribution across ratings
    assert abs(a["output_power"].mean() - b["output_power"].mean()) < 0.02


def test_platform_step_and_advice():
    d = generate("pump_drive_5hp", days=8, seed=3, n_true=3, n_near=1)
    P = d["profile"]; e = window_ends(d["pu"].index); y, m = targets(P, d["labels"], e)
    model = EarlyWarningXGB(P.alarm_codes, n_estimators=30).fit(pu_features(P, d["pu"], e), y, m)
    out = Platform(Profile.load("pump_drive_10hp"), model).step(d["raw"].iloc[600:720])
    assert out["state"] in {"NORMAL", "WARNING", "ALARM"}
    adv = AdvisorAgent().advise("DRV-05")
    assert adv["steps"] and adv["source"]
