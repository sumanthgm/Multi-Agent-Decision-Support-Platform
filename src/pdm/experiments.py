"""All Phase II experiments in one place, used by scripts/run_experiments.py and the Colab notebook.

E1  Multi-fault early warning on the training rating (unseen seeds)
E2  Same models on unseen power ratings (profile swap only, no retraining)
E3  Forecast Agent: 60-minute forecast error and time-to-trip on thermal alarms
E4  Advisor Agent: correct manual entry for each alarm event vs README-style raw-value RAG
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .agents.advisor import AdvisorAgent
from .agents.early_warning import EarlyWarningXGB
from .agents.forecast import forecast, minutes_to_limit
from .baseline.aspire_transformer import TORCH_OK, AspireTransformer
from .eval.metrics import evaluate, summarize
from .features import pu_features, raw_features, sequences, targets, window_ends
from .twin import generate

ROOT = Path(__file__).resolve().parents[2]
METHOD_LABELS = {
    "ours_pu_xgb": "Ours: per-unit + limit features, XGBoost",
    "ablation_raw_xgb": "Ablation: raw features, XGBoost",
    "aspire_transformer_raw": "ASPIRE-style Transformer (raw, z-scored)",
    "aspire_transformer_pu": "ASPIRE-style Transformer on per-unit signals",
}
NAMES = {  # raw column -> words used by the README-style RAG query
    "ac_v": ("grid voltage", "V"),
    "ac_f": ("grid frequency", "Hz"),
    "p_out": ("output power", "W"),
    "p_pv": ("PV power", "W"),
    "v_pv": ("PV voltage", "V"),
    "r_iso": ("insulation resistance", "kOhm"),
    "v_bat": ("battery voltage", "V"),
    "soc": ("state of charge", "%"),
    "t_bat": ("battery temperature", "C"),
    "t_hs": ("heatsink temperature", "C"),
    "dg_f": ("generator frequency", "Hz"),
    "dg_on": ("generator on", ""),
    "g": ("irradiance", "W/m2"),
    "v_dc": ("DC bus voltage", "V"),
    "i_out": ("motor current", "A"),
    "f_out": ("output frequency", "Hz"),
    "level": ("tank level", "%"),
    "i_imb": ("current imbalance", "%"),
    "flow": ("flow", "m3/h"),
}


def load_config(quick: bool = False) -> dict:
    cfg = yaml.safe_load((ROOT / "configs" / "experiment.yaml").read_text())
    if quick:
        for k, v in cfg["quick"].items():
            cfg[k] = v
    return cfg


class DataCache:
    def __init__(self):
        self._d = {}

    def get(self, name: str, days: int, seed: int, n_true: int, n_near: int) -> dict:
        key = (name, days, seed, n_true, n_near)
        if key not in self._d:
            self._d[key] = generate(name, days, seed, n_true=n_true, n_near=n_near)
        return self._d[key]


def _inputs(method: str, d: dict, ends: np.ndarray):
    P = d["profile"]
    if method == "ours_pu_xgb":
        return pu_features(P, d["pu"], ends)
    if method == "ablation_raw_xgb":
        return raw_features(d["raw"], ends)
    if method == "aspire_transformer_raw":
        return sequences(d["raw"], ends)
    if method == "aspire_transformer_pu":
        return sequences(d["pu"], ends)
    raise ValueError(method)


def run_early_warning(cfg: dict, cache: DataCache, out: Path, log=print):
    rows, preds = [], {}
    methods = [m for m in cfg["methods"] if TORCH_OK or "transformer" not in m]
    if len(methods) < len(cfg["methods"]):
        log("PyTorch not installed: ASPIRE-style Transformer baselines skipped.")
    tr_cfg, te_cfg, tf_cfg = cfg["train"], cfg["test"], cfg["transformer"]
    (out / "models").mkdir(parents=True, exist_ok=True)
    for asset, spec in cfg["assets"].items():
        dtr = cache.get(spec["train"], tr_cfg["days"], tr_cfg["seed"], tr_cfg["n_true"], tr_cfg["n_near"])
        P = dtr["profile"]
        codes = P.alarm_codes
        H = {c: P.alarms[c].warn_minutes for c in codes}
        e = window_ends(dtr["pu"].index)
        y, m = targets(P, dtr["labels"], e)
        for method in methods:
            t0 = time.time()
            if "xgb" in method:
                model = EarlyWarningXGB(codes).fit(_inputs(method, dtr, e), y, m)
            else:
                s = tf_cfg["train_stride"]
                model = AspireTransformer(codes, epochs=tf_cfg["epochs"], batch=tf_cfg["batch"], lr=tf_cfg["lr"]).fit(
                    _inputs(method, dtr, e[::s]), y[::s], m[::s]
                )
            log(f"[{asset}] trained {method} in {time.time() - t0:.0f}s")
            if method == "ours_pu_xgb":
                with open(out / "models" / f"{spec['train']}_ours.pkl", "wb") as f:
                    pickle.dump(model, f)
            for name in spec["test"]:
                for seed in te_cfg["seeds"]:
                    dte = cache.get(name, te_cfg["days"], seed, te_cfg["n_true"], te_cfg["n_near"])
                    et = window_ends(dte["pu"].index)
                    yt, mt = targets(dte["profile"], dte["labels"], et)
                    Xt = _inputs(method, dte, et)
                    prob, pred = model.predict_proba(Xt), model.predict(Xt)
                    preds[(method, name, seed)] = (prob, pred, et)
                    res = evaluate(codes, yt, mt, pred, prob, dte["labels"], et, H)
                    res.insert(0, "seed", seed)
                    res.insert(0, "rating", name)
                    res.insert(0, "asset", asset)
                    res.insert(0, "method", method)
                    rows.append(res)
    per_alarm = pd.concat(rows, ignore_index=True)
    per_alarm.to_csv(out / "e1_e2_per_alarm.csv", index=False)

    summ = []
    for (method, asset, rating, seed), g in per_alarm.groupby(["method", "asset", "rating", "seed"]):
        s = summarize(g)
        s.update(method=method, asset=asset, rating=rating, seed=seed)
        summ.append(s)
    summ = pd.DataFrame(summ)
    agg = (
        summ.groupby(["asset", "rating", "method"])
        .agg(
            macro_f1=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            macro_pr_auc=("macro_pr_auc", "mean"),
            event_recall=("event_recall", "mean"),
            median_lead_min=("median_lead_min", "mean"),
            false_alerts_per_day=("false_alerts_per_day", "mean"),
        )
        .reset_index()
    )
    agg["trained_on"] = agg["rating"].isin([s["train"] for s in cfg["assets"].values()])
    agg.to_csv(out / "e1_e2_summary.csv", index=False)
    return per_alarm, agg, preds


def run_forecast(cfg: dict, cache: DataCache, out: Path, log=print) -> pd.DataFrame:
    fc = cfg["forecast"]
    Hh = fc["horizon_min"]
    te = cfg["test"]
    methods = list(fc["methods"])
    if "chronos" in methods:
        try:
            from .agents.forecast import _chronos

            _chronos()
        except Exception as exc:
            log(f"Chronos unavailable ({type(exc).__name__}); skipped.")
            methods.remove("chronos")
    targets_by_asset = {
        "hybrid_inverter": ["heatsink_temp", "battery_soc", "output_power"],
        "pump_drive": ["heatsink_temp", "tank_level", "output_power"],
    }
    thermal = {"hybrid_inverter": "INV-11", "pump_drive": "DRV-04"}
    rows, ttt = [], []
    for asset, spec in cfg["assets"].items():
        d = cache.get(spec["train"], te["days"], te["seeds"][0], te["n_true"], te["n_near"])
        pu = d["pu"]
        origins = np.arange(1440 + 60, len(pu) - Hh, 60)
        for sig in targets_by_asset[asset]:
            x = pu[sig].to_numpy()
            for method in methods:
                org = origins[:: max(1, len(origins) // fc["chronos_max_origins"])] if method == "chronos" else origins
                err = [np.mean(np.abs(forecast(x[:o], Hh, method) - x[o : o + Hh])) for o in org]
                rows.append(dict(asset=asset, signal=sig, method=method, mae_pu=float(np.mean(err)), origins=len(org)))
        a = d["profile"].alarms[thermal[asset]]
        on = np.flatnonzero(d["labels"][f"{a.code}_onset"].to_numpy())
        for method in methods:
            for t_o in on:
                o = t_o - 30
                if o < 120:
                    continue
                pred = minutes_to_limit(forecast(pu[a.signal].to_numpy()[:o], Hh, method), a)
                ttt.append(
                    dict(
                        asset=asset,
                        alarm=a.code,
                        method=method,
                        true_min=30,
                        predicted_min=pred,
                        caught=pred is not None,
                        abs_err_min=abs(pred - 30) if pred is not None else np.nan,
                    )
                )
    df = pd.DataFrame(rows)
    df.to_csv(out / "e3_forecast_mae.csv", index=False)
    tt = pd.DataFrame(ttt)
    tt.to_csv(out / "e3_time_to_trip.csv", index=False)
    return df, tt


def run_advisor(cfg: dict, cache: DataCache, preds: dict, out: Path) -> pd.DataFrame:
    adv = AdvisorAgent()
    te = cfg["test"]
    rows = []
    for asset, spec in cfg["assets"].items():
        for name in spec["test"]:
            for seed in te["seeds"]:
                d = cache.get(name, te["days"], seed, te["n_true"], te["n_near"])
                P = d["profile"]
                codes = P.alarm_codes
                prob, pred, et = preds[("ours_pu_xgb", name, seed)]
                for code in codes:
                    for t_o in np.flatnonzero(d["labels"][f"{code}_onset"].to_numpy()):
                        H = P.alarms[code].warn_minutes
                        win = np.flatnonzero((et >= t_o - H) & (et < t_o))
                        if not len(win):
                            continue
                        fired = win[pred[win].any(axis=1)]
                        i = fired[0] if len(fired) else win[-1]
                        top = [codes[k] for k in np.argsort(-prob[i])[:3]]
                        snap = d["raw"].iloc[et[i]]
                        text = ", ".join(f"{NAMES[c][0]} {snap[c]:.2f} {NAMES[c][1]}" for c in d["raw"].columns)
                        base = adv.search(text, k=3)
                        rows.append(
                            dict(
                                asset=asset,
                                rating=name,
                                seed=seed,
                                alarm=code,
                                warned=bool(len(fired)),
                                ours_hit1=top[0] == code,
                                ours_hit3=code in top,
                                readme_rag_hit1=base[0] == code,
                                readme_rag_hit3=code in base,
                                has_kb_entry=code in adv.kb,
                            )
                        )
    df = pd.DataFrame(rows)
    df.to_csv(out / "e4_advisor.csv", index=False)
    return df


def run_all(quick: bool = False, out_dir: str | Path | None = None, log=print) -> dict:
    cfg = load_config(quick)
    out = Path(out_dir) if out_dir else ROOT / ("results_quick" if quick else "results")
    out.mkdir(parents=True, exist_ok=True)
    cache = DataCache()
    t0 = time.time()
    per_alarm, agg, preds = run_early_warning(cfg, cache, out, log)
    fc, tt = run_forecast(cfg, cache, out, log)
    adv = run_advisor(cfg, cache, preds, out)
    from .eval.report import write_report

    write_report(cfg, out, per_alarm, agg, fc, tt, adv, cache, preds)
    meta = dict(quick=quick, runtime_s=round(time.time() - t0), torch=TORCH_OK, config=cfg)
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    log(f"Done in {meta['runtime_s']} s. Report: {out / 'report.md'}")
    return dict(per_alarm=per_alarm, summary=agg, forecast=fc, time_to_trip=tt, advisor=adv, out=out)
