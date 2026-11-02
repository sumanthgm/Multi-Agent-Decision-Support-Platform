"""Figures and the markdown results report (results/report.md)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHOD_SHORT = {"ours_pu_xgb": "Ours (per-unit XGB)", "ablation_raw_xgb": "Raw XGB (ablation)",
                "aspire_transformer_raw": "ASPIRE-style Transformer (raw)",
                "aspire_transformer_pu": "ASPIRE-style Transformer (per-unit)"}
COLORS = {"ours_pu_xgb": "#2a6fdb", "ablation_raw_xgb": "#9aa5b1",
          "aspire_transformer_raw": "#e07b39", "aspire_transformer_pu": "#f2b880"}


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt.format(v)
        return str(v)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def _fig_cross_rating(agg: pd.DataFrame, cfg: dict, out: Path):
    assets = list(agg["asset"].unique())
    fig, axes = plt.subplots(1, len(assets), figsize=(6 * len(assets), 4), squeeze=False)
    for ax, asset in zip(axes[0], assets):
        g = agg[agg.asset == asset]
        import re
        ratings = sorted(dict.fromkeys(g.rating), key=lambda r: int(re.findall(r"(\d+)(?:kw|hp)$", r)[0]))
        trained = {s_["train"] for s_ in cfg["assets"].values()}
        methods = [m for m in METHOD_SHORT if m in set(g.method)]
        w = 0.8 / len(methods)
        for k, mth in enumerate(methods):
            gm = g[g.method == mth].set_index("rating").reindex(ratings)
            ax.bar(np.arange(len(ratings)) + k * w, gm.macro_f1, w, yerr=gm.macro_f1_std.fillna(0),
                   color=COLORS[mth], label=METHOD_SHORT[mth], capsize=3)
        ax.set_xticks(np.arange(len(ratings)) + w * (len(methods) - 1) / 2)
        ax.set_xticklabels([r.split("_")[-1].replace("kw", " kW").replace("hp", " HP") + (" (trained)" if r in trained else "") for r in ratings])
        ax.set_ylim(0, 1); ax.set_ylabel("Macro-F1 (early warning)"); ax.set_title(asset.replace("_", " "))
        ax.grid(axis="y", alpha=0.3)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=9, frameon=False)
    fig.suptitle("Trained on the 5 kW / 5 HP unit; other ratings use only a new profile")
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(out / "fig_cross_rating.png", dpi=150); plt.close(fig)


def _fig_lead(per_alarm: pd.DataFrame, cfg: dict, out: Path):
    trained = [s["train"] for s in cfg["assets"].values()]
    g = per_alarm[(per_alarm.method == "ours_pu_xgb") & (per_alarm.rating.isin(trained))]
    g = g.groupby("alarm")[["event_recall", "median_lead_min"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.bar(g.alarm, g.median_lead_min, color="#2a6fdb")
    for i, r in g.iterrows():
        ax.text(i, (r.median_lead_min if not np.isnan(r.median_lead_min) else 0) + 2,
                f"{r.event_recall:.0%}", ha="center", fontsize=8)
    ax.set_ylabel("Median lead time (min)"); ax.set_title("Ours: lead time per alarm (label = events warned)")
    plt.xticks(rotation=45); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "fig_lead_time.png", dpi=150); plt.close(fig)


def _fig_example(cfg, cache, preds, out: Path):
    spec = cfg["assets"]["hybrid_inverter"]; te = cfg["test"]
    d = cache.get(spec["train"], te["days"], te["seeds"][0], te["n_true"], te["n_near"])
    P = d["profile"]; code = "INV-11"; a = P.alarms[code]
    on = np.flatnonzero(d["labels"][f"{code}_onset"].to_numpy())
    if not len(on):
        return
    t_o = on[0]; lo, hi = max(0, t_o - 240), min(len(d["pu"]), t_o + 90)
    prob, pred, et = preds[("ours_pu_xgb", spec["train"], te["seeds"][0])]
    j = P.alarm_codes.index(code)
    sel = (et >= lo) & (et < hi)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    t = (np.arange(lo, hi) - t_o)
    ax.plot(t, d["pu"][a.signal].to_numpy()[lo:hi], color="#333", label="heatsink temp (pu of trip)")
    ax.axhline(a.limit_pu, color="#c0392b", ls="--", label="trip limit")
    ax.axvspan(-a.warn_minutes, 0, color="#f5b041", alpha=0.25, label=f"warning horizon ({a.warn_minutes} min)")
    ax2 = ax.twinx(); ax2.plot(et[sel] - t_o, prob[sel, j], color="#2a6fdb", marker=".", label="predicted probability")
    ax2.set_ylim(0, 1.05); ax2.set_ylabel("P(alarm within horizon)")
    ax.set_xlabel("minutes relative to alarm onset"); ax.set_ylabel("pu")
    ax.set_title(f"{code} {a.name}: warning before the trip (5 kW test unit)")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(out / "fig_example_INV-11.png", dpi=150); plt.close(fig)


def write_report(cfg, out: Path, per_alarm, agg, fc, tt, adv, cache, preds):
    _fig_cross_rating(agg, cfg, out); _fig_lead(per_alarm, cfg, out); _fig_example(cfg, cache, preds, out)
    trained = [s["train"] for s in cfg["assets"].values()]
    n_seeds = len(cfg["test"]["seeds"])
    lines = ["# Phase II results: rating-aware multi-agent fault prediction", "",
             f"Test set: {cfg['test']['days']} days per unit, {n_seeds} random seed(s) per rating, unseen weather and fault timings. "
             f"Training: one {cfg['train']['days']}-day unit per asset type (5 kW inverter, 5 HP pump drive). "
             "All data is from the telemetry twin (see Limitations).", ""]

    tf = cfg["transformer"]
    lines += ["## E1 - Multi-fault early warning (trained rating, unseen seeds)", "",
              "All methods see the same windows (60 min history, a prediction every 5 min) and tune one threshold per alarm "
              f"on the last 20% of training. The ASPIRE-style Transformer is trained for {tf['epochs']} epochs on every "
              f"{tf['train_stride']}-th window with class-weighted loss. Means over seeds; std across seeds.", ""]
    t = agg[agg.rating.isin(trained)].copy()
    t["method"] = t.method.map(METHOD_SHORT)
    lines += [md_table(t[["asset", "method", "macro_f1", "macro_f1_std", "macro_pr_auc", "event_recall",
                          "median_lead_min", "false_alerts_per_day"]].round(3)), ""]

    lines += ["## E2 - Unseen power ratings (profile swap only, no retraining)", ""]
    size = {}
    for s in cfg["assets"].values():
        for r in s["test"]:
            size[r] = "trained rating" if r == s["train"] else ("smaller rating" if r.endswith(("3kw", "3hp")) else "larger rating")
    a2 = agg.assign(size=agg.rating.map(size))
    piv = a2.pivot_table(index=["asset", "method"], columns="size", values="macro_f1").reset_index()
    piv = piv[["asset", "method", "trained rating", "smaller rating", "larger rating"]]
    piv["drop_smaller"] = piv["trained rating"] - piv["smaller rating"]
    piv["drop_larger"] = piv["trained rating"] - piv["larger rating"]
    piv["method"] = piv.method.map(METHOD_SHORT)
    lines += ["Macro-F1. Smaller = 3 kW / 3 HP, larger = 10 kW / 10 HP. Drop = trained minus unseen (lower is better).", "",
              md_table(piv.round(3)), "", "![cross rating](fig_cross_rating.png)", ""]

    lines += ["## Per-alarm results for our method (trained rating, mean over seeds)", ""]
    pa = per_alarm[(per_alarm.method == "ours_pu_xgb") & per_alarm.rating.isin(trained)]
    pa = pa.groupby(["asset", "alarm"])[["precision", "recall", "f1", "pr_auc", "events", "event_recall",
                                         "median_lead_min", "false_alerts_per_day"]].mean().reset_index()
    lines += [md_table(pa.round(3)), "", "![lead time](fig_lead_time.png)", "", "![example](fig_example_INV-11.png)", ""]

    lines += ["## E3 - Forecast Agent (60-minute horizon)", "", "Mean absolute error in per-unit:", ""]
    lines += [md_table(fc.pivot_table(index=["asset", "signal"], columns="method", values="mae_pu").reset_index().round(4)), ""]
    if len(tt):
        tsum = tt.groupby(["asset", "alarm", "method"]).agg(events=("caught", "size"), caught_rate=("caught", "mean"),
                                                            abs_err_min=("abs_err_min", "mean")).reset_index()
        lines += ["Time to trip, forecast made 30 minutes before each thermal alarm (caught = forecast crosses the limit within 60 min):", "",
                  md_table(tsum.round(3)), ""]

    lines += ["## E4 - Advisor Agent: correct manual entry per alarm event", ""]
    a4 = adv.groupby("asset")[["ours_hit1", "ours_hit3", "readme_rag_hit1", "readme_rag_hit3"]].mean().reset_index()
    a4["events"] = adv.groupby("asset").size().values
    lines += [md_table(a4.round(3)), "",
              "Ours routes the Early-Warning agent's top alarm codes to their knowledge-base entries. "
              "README-style RAG queries the knowledge base with the raw sensor values, as the reference project does.", ""]

    lines += ["## Scope compared with the ASPIRE baseline", "",
              "| Item | ASPIRE (as reproduced in Phase I) | This framework |", "| --- | --- | --- |",
              "| Assets | Metro air compressor (MetroPT-3) | Hybrid inverter, solar pump drive |",
              f"| Alarm types predicted | 1 failure category | {sum(len(cache.get(s['train'], cfg['train']['days'], cfg['train']['seed'], cfg['train']['n_true'], cfg['train']['n_near'])['profile'].alarms) for s in cfg['assets'].values())} |",
              "| Power ratings | 1 | 3 per asset type, by profile |", "| Fix guidance | None | Cited checklist per alarm |", "",
              "## Limitations", "",
              "- Faults are injected into synthetic healthy telemetry; thresholds are placeholders until replaced with datasheet values.",
              "- Per-unit features remove rating scale by design in this twin; real cross-rating data is needed to confirm the gain.",
              "- Validate on real labelled faults (GPVS-Faults, inverter-driven PMSM dataset) with `scripts/validate_real.py` before claiming field performance.",
              "- The knowledge base is authored from general practice; replace each source with the manual page of the reference device.",
              "- ASPIRE-style baseline re-implements the Prediction Agent's Transformer only, not the full ASPIRE system."]
    (out / "report.md").write_text("\n".join(lines))
