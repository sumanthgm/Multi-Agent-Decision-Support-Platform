"""Streamlit operator dashboard for the agentic PdM platform.

Start the API first, then:  streamlit run app/dashboard.py
Pick the asset in the sidebar and press Start under "Live feed" to stream twin telemetry,
or post your own rows with  python scripts/replay.py --profile <name> --mode api.
"""

from __future__ import annotations

import threading

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="Agentic PdM", page_icon="⚡", layout="wide")

ASSET_TYPES = {"hybrid_inverter": "⚡ Hybrid inverter", "pump_drive": "💧 Solar pump drive"}
# pu role -> (label, unit, group); the raw column for a role comes from the profile's signal map
SIGNALS = {
    "hybrid_inverter": {
        "grid_voltage": ("Grid voltage", "V", "Grid"),
        "grid_frequency": ("Grid frequency", "Hz", "Grid"),
        "output_power": ("AC output power", "W", "Grid"),
        "pv_power": ("PV power", "W", "PV array"),
        "pv_voltage": ("PV voltage", "V", "PV array"),
        "insulation": ("PV insulation", "kΩ", "PV array"),
        "irradiance": ("Irradiance", "W/m²", "PV array"),
        "battery_voltage": ("Battery voltage", "V", "Battery"),
        "battery_soc": ("Battery SOC", "%", "Battery"),
        "battery_temp": ("Battery temperature", "°C", "Battery"),
        "heatsink_temp": ("Heatsink temperature", "°C", "Thermal"),
        "dg_frequency": ("Generator frequency", "Hz", "Generator"),
        "dg_on": ("Generator running", "", "Generator"),
    },
    "pump_drive": {
        "dc_voltage": ("DC-bus voltage", "V", "Electrical"),
        "output_current": ("Motor current", "A", "Electrical"),
        "output_power": ("Output power", "W", "Electrical"),
        "current_imbalance": ("Phase imbalance", "%", "Electrical"),
        "speed": ("Output frequency", "Hz", "Motor & pump"),
        "flow": ("Flow", "m³/h", "Motor & pump"),
        "tank_level": ("Tank level", "%", "Motor & pump"),
        "heatsink_temp": ("Heatsink temperature", "°C", "Thermal"),
        "irradiance": ("Irradiance", "W/m²", "Solar"),
    },
}
KPIS = {
    "hybrid_inverter": [
        "grid_voltage",
        "grid_frequency",
        "output_power",
        "pv_power",
        "battery_soc",
        "battery_voltage",
        "heatsink_temp",
        "insulation",
    ],
    "pump_drive": [
        "dc_voltage",
        "output_current",
        "output_power",
        "speed",
        "flow",
        "tank_level",
        "heatsink_temp",
        "current_imbalance",
    ],
}
DERIVED = {"grid_freq_dev": "grid_frequency", "dg_freq_dev": "dg_frequency", "load_ratio": "output_power"}
COLOURS = {"ALARM": "#ff4d6d", "WARNING": "#ffb020", "NORMAL": "#2ecc71", "IDLE": "#64748b"}
ACCENT = "#4cc9f0"
VIEWS = ["📈 Live telemetry", "🚨 Alarms & forecast", "🛠 Maintenance advisor", "⚙️ Asset profile"]

CSS = """<style>
.stApp {background: radial-gradient(1100px 520px at 0% -10%, rgba(76,201,240,.16), transparent 60%), radial-gradient(900px 480px at 100% 0%, rgba(167,139,250,.14), transparent 60%), linear-gradient(180deg, #0a1020 0%, #070b16 100%);}
[data-testid="stSidebar"] {background: rgba(10,16,32,.88); border-right: 1px solid rgba(255,255,255,.06);}
[data-testid="stHeader"] {background: transparent;}
.block-container {padding-top: 2.2rem;}
.hero {display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px; padding:18px 22px; border-radius:18px; background:linear-gradient(135deg, rgba(76,201,240,.12), rgba(167,139,250,.10)); border:1px solid rgba(255,255,255,.08); margin-bottom:14px;}
.hero h1 {font-size:1.65rem; margin:0; padding:0; color:#eef4ff;}
.hero .sub {color:#9fb0c8; font-size:.92rem; margin-top:4px;}
.badge {display:inline-block; padding:6px 18px; border-radius:999px; font-weight:800; letter-spacing:.06em; color:#0a1020; background:var(--c); box-shadow:0 0 14px var(--c);}
.badge.pulse {animation: pulse 1.2s infinite;}
@keyframes pulse {0%,100% {box-shadow:0 0 4px var(--c);} 50% {box-shadow:0 0 28px var(--c);}}
.meta {color:#9fb0c8; font-size:.82rem; text-align:right; margin-top:8px;}
.grid {display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:12px; margin:4px 0 16px;}
@media (max-width: 1000px) {.grid {grid-template-columns:repeat(2, minmax(0, 1fr));}}
.kpi {background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08); border-left:4px solid var(--c); border-radius:14px; padding:12px 14px 8px;}
.kpi .l {font-size:.72rem; color:#9fb0c8; text-transform:uppercase; letter-spacing:.06em;}
.kpi .v {font-size:1.45rem; font-weight:700; color:#eef4ff; line-height:1.3;}
.kpi .v small {font-size:.8rem; color:#9fb0c8; font-weight:500; margin-left:4px;}
.kpi .f {font-size:.72rem; color:#8193ad; display:flex; justify-content:space-between; gap:6px;}
.kpi svg {width:100%; height:34px; display:block; margin:2px 0;}
.card {background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08); border-left:4px solid var(--c); border-radius:14px; padding:12px 16px; margin-bottom:10px;}
.card .t {font-weight:700; color:#eef4ff;}
.card .code {display:inline-block; font-family:monospace; font-size:.78rem; padding:1px 8px; border-radius:6px; background:var(--c); color:#0a1020; margin-right:8px; font-weight:700;}
.card .d {color:#9fb0c8; font-size:.85rem; margin-top:4px;}
.bar {height:6px; border-radius:4px; background:rgba(255,255,255,.08); margin-top:8px; overflow:hidden;}
.bar > div {height:100%; background:var(--c);}
.empty {text-align:center; padding:40px 20px; color:#9fb0c8; border:1px dashed rgba(255,255,255,.15); border-radius:16px; margin-top:8px;}
.empty b {color:#eef4ff; font-size:1.1rem;}
</style>"""


# ------------------------------------------------------------------ API calls
@st.cache_data(ttl=1, show_spinner=False)
def get(api: str, path: str, **params):
    r = requests.get(f"{api}{path}", params=params, timeout=120)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=300, show_spinner=False)
def get_static(api: str, path: str, **params):
    """Profiles and knowledge-base entries change only when the API restarts."""
    return get(api, path, **params)


def post(api: str, path: str, **params):
    r = requests.post(f"{api}{path}", params=params, timeout=60)
    r.raise_for_status()
    get.clear()
    return r.json()


# ------------------------------------------------------------------ formatting
def rating_label(name: str) -> str:
    size = name.rsplit("_", 1)[-1]
    return f"{size[:-2]} {size[-2:].replace('kw', 'kW').replace('hp', 'HP')}"


def fmt(v, unit: str) -> tuple[str, str]:
    if v is None or pd.isna(v):
        return "–", unit
    if unit == "":
        return ("ON" if v > 0.5 else "OFF"), ""
    if unit == "W" and abs(v) >= 1000:
        return f"{v / 1000:,.2f}", "kW"
    return f"{v:,.{2 if unit == 'Hz' or abs(v) < 10 else 1}f}", unit


def limit_text(a: dict, unit: str) -> str:
    lims = a["raw_limits"]
    if len(lims) == 2:
        return f"{a['code']}: outside {fmt(lims[0], unit)[0]}–{' '.join(fmt(lims[1], unit))}"
    op = ">" if a["direction"] == "above" else "<"
    if lims:
        return f"{a['code']}: {op} {' '.join(fmt(lims[0], unit))}"
    return f"{a['code']}: {a['signal']} {op} {a['limit_pu']:g} pu"


def sparkline(vals: list, colour: str, w: int = 160, h: int = 34) -> str:
    v = [x for x in vals if x is not None and not pd.isna(x)]
    if len(v) < 2:
        return f'<svg viewBox="0 0 {w} {h}"></svg>'
    lo, hi = min(v), max(v)
    span = (hi - lo) or 1.0
    pts = " ".join(f"{i * w / (len(v) - 1):.1f},{h - 3 - (x - lo) / span * (h - 6):.1f}" for i, x in enumerate(v))
    return (
        f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none"><polyline points="{pts}" fill="none" '
        f'stroke="{colour}" stroke-width="1.8" vector-effect="non-scaling-stroke"/></svg>'
    )


def style(fig: go.Figure, title: str, height: int = 230) -> go.Figure:
    grid = "rgba(255,255,255,.06)"
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=40, b=8),
        showlegend=False,
        hovermode="x unified",
        title=dict(text=title, font=dict(size=14, color="#dbe6f7"), x=0.01),
        paper_bgcolor="rgba(255,255,255,0.03)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9fb0c8"),
        xaxis=dict(gridcolor=grid),
        yaxis=dict(gridcolor=grid),
    )
    return fig


def chart(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, key=key, config={"displayModeBar": False})


# ------------------------------------------------------------------ helpers on profile + status
def role_columns(prof: dict) -> dict[str, str]:
    return {role: col for col, role in prof["signals"].items()}


def alarms_by_column(prof: dict) -> dict[str, list[dict]]:
    """Raw column -> alarms watching it (derived signals map to the raw column they come from)."""
    col_of, out = role_columns(prof), {}
    for a in prof["alarms"]:
        col = a["raw_column"] or col_of.get(DERIVED.get(a["signal"], a["signal"]))
        out.setdefault(col, []).append(a)
    return out


def alarm_state(code: str, status: dict) -> str:
    if code in {a["code"] for a in status.get("alarms", [])}:
        return "ALARM"
    if code in {w["code"] for w in status.get("warnings", [])}:
        return "WARNING"
    return "NORMAL"


def worst(states: list[str]) -> str:
    return next((s for s in ("ALARM", "WARNING") if s in states), "NORMAL")


def eng(entry: dict, pu: float, t: str) -> tuple[str, str]:
    """A pu value of an alarm signal in engineering units when the profile can scale it back."""
    if entry.get("scale"):
        return fmt(pu * entry["scale"], SIGNALS[t].get(entry["signal"], ("", "pu", ""))[1])
    return f"{pu:.3f}", "pu"


# ------------------------------------------------------------------ page sections
def hero(prof: dict, status: dict) -> None:
    state = status.get("state") or {"no data": "NO DATA", "warming_up": "WARMING UP"}.get(status.get("status"), "–")
    c = COLOURS.get(state, COLOURS["IDLE"])
    feed = status.get("feed") or {}
    feed_txt = f"📡 live feed · {feed['speed']:g} min/s" if feed.get("running") else "📡 feed stopped"
    st.markdown(
        f'<div class="hero"><div><h1>Agentic predictive maintenance</h1>'
        f'<div class="sub">{ASSET_TYPES[prof["asset_type"]]} · {rating_label(prof["name"])} · {prof["name"]}</div></div>'
        f'<div><span class="badge{" pulse" if state == "ALARM" else ""}" style="--c:{c}">{state}</span>'
        f'<div class="meta">🕒 {status.get("time", "–")} · {status.get("rows", 0)} rows · {feed_txt}</div></div></div>',
        unsafe_allow_html=True,
    )


def kpi_cards(prof: dict, hist: pd.DataFrame, status: dict) -> None:
    t, col_of, by_col = prof["asset_type"], role_columns(prof), alarms_by_column(prof)
    cards = []
    for role in KPIS[t]:
        col = col_of.get(role)
        if col not in hist:
            continue
        label, unit, _ = SIGNALS[t][role]
        watch = sorted(
            by_col.get(col, []), key=lambda a: ["ALARM", "WARNING", "NORMAL"].index(alarm_state(a["code"], status))
        )
        state = worst([alarm_state(a["code"], status) for a in watch])
        c = COLOURS[state]
        v, u = fmt(hist[col].iloc[-1], unit)
        foot = limit_text(watch[0], unit) if watch else "monitored"
        cards.append(
            f'<div class="kpi" style="--c:{c}"><div class="l">{label}</div>'
            f'<div class="v">{v}<small>{u}</small></div>'
            f"{sparkline(hist[col].tail(90).tolist(), ACCENT if state == 'NORMAL' else c)}"
            f'<div class="f"><span>{foot}</span><span>{state.title()}</span></div></div>'
        )
    st.markdown('<div class="grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def view_telemetry(prof: dict, hist: pd.DataFrame, status: dict) -> None:
    t, col_of, by_col = prof["asset_type"], role_columns(prof), alarms_by_column(prof)
    meta = SIGNALS[t]
    groups = list(dict.fromkeys(g for _, _, g in meta.values()))
    chosen = st.pills("Signal groups", groups, selection_mode="multi", default=groups, key=f"groups_{t}") or []
    shown = [(role, col_of[role]) for role, (_, _, g) in meta.items() if g in chosen and col_of.get(role) in hist]
    if not shown:
        st.info("Pick at least one signal group above.")
        return
    cols = st.columns(2)
    for i, (role, col) in enumerate(shown):
        label, unit, _ = meta[role]
        watch = by_col.get(col, [])
        state = worst([alarm_state(a["code"], status) for a in watch])
        fig = go.Figure(
            go.Scatter(
                x=hist.index,
                y=hist[col],
                mode="lines",
                line=dict(color=ACCENT if state == "NORMAL" else COLOURS[state], width=2),
                hovertemplate=f"%{{y:.4g}} {unit}<extra></extra>",
            )
        )
        for a in watch:
            for lim in a["raw_limits"]:
                fig.add_hline(
                    y=lim,
                    line=dict(color=COLOURS["ALARM"], width=1, dash="dash"),
                    annotation_text=f"{a['code']} limit",
                    annotation_position="top left",
                    annotation_font=dict(color="#ff8fa3", size=10),
                )
        with cols[i % 2]:
            chart(style(fig, f"{label} ({unit})" if unit else label), key=f"sig_{col}")


def alarm_card(code: str, title: str, lines: list[str], c: str, bar: float | None = None) -> str:
    bar_html = f'<div class="bar"><div style="width:{bar * 100:.0f}%"></div></div>' if bar is not None else ""
    return (
        f'<div class="card" style="--c:{c}"><div class="t"><span class="code">{code}</span>{title}</div>'
        + "".join(f'<div class="d">{x}</div>' for x in lines)
        + bar_html
        + "</div>"
    )


def view_alarms(prof: dict, status: dict, fc: dict) -> None:
    t = prof["asset_type"]
    entry = {a["code"]: a for a in fc["alarms"]}
    left, right = st.columns(2)
    with left:
        st.markdown("#### 🔴 Active alarms")
        if not status.get("alarms"):
            st.success("No active alarms")
        for a in status.get("alarms", []):
            e = entry[a["code"]]
            st.markdown(
                alarm_card(
                    a["code"],
                    a["name"],
                    [
                        f"Now <b>{' '.join(eng(e, a['value_pu'], t))}</b> · limit {' '.join(eng(e, a['limit_pu'], t))}",
                        f"Signal {e['signal']} is past its limit ({e['direction']})",
                    ],
                    COLOURS["ALARM"],
                ),
                unsafe_allow_html=True,
            )
    with right:
        st.markdown("#### 🟠 Predicted alarms")
        if not status.get("warnings"):
            st.success("No alarms predicted in the warning horizon")
        for w in status.get("warnings", []):
            ttt = w["forecast_minutes_to_trip"]
            st.markdown(
                alarm_card(
                    w["code"],
                    w["name"],
                    [
                        f"Probability <b>{w['probability']:.0%}</b> of tripping within {w['horizon_min']} min",
                        f"Trend forecast: trips in <b>~{ttt} min</b>"
                        if ttt
                        else "Trend forecast: no crossing in the next hour",
                    ],
                    COLOURS["WARNING"],
                    bar=w["probability"],
                ),
                unsafe_allow_html=True,
            )

    st.markdown("#### 📋 Alarm watchlist")
    rows = []
    for e in fc["alarms"]:
        state = alarm_state(e["code"], status)
        if state == "NORMAL" and not e["condition_met"]:
            state = "IDLE"
        unit = eng(e, e["value_pu"], t)[1]
        rows.append(
            {
                "Status": {"ALARM": "🔴 Alarm", "WARNING": "🟠 Warning", "NORMAL": "🟢 OK", "IDLE": "⚪ Idle"}[state],
                "Code": e["code"],
                "Alarm": e["name"],
                "Now": eng(e, e["value_pu"], t)[0],
                "Limit": f"{'>' if e['direction'] == 'above' else '<'} {eng(e, e['limit_pu'], t)[0]}",
                "Unit": unit,
                "Trip forecast": f"~{e['minutes_to_trip']} min" if e["minutes_to_trip"] else "—",
                "_rank": ["ALARM", "WARNING", "NORMAL", "IDLE"].index(state),
            }
        )
    df = pd.DataFrame(rows).sort_values("_rank", kind="stable").drop(columns="_rank")
    st.dataframe(
        df,
        hide_index=True,
        width="stretch",
        column_config={"Trip forecast": st.column_config.TextColumn(help="Trend forecast over the next hour")},
    )
    st.caption("⚪ Idle: the alarm's operating condition (daylight, running, generator on…) is not met right now.")

    st.markdown("#### 🔮 Forecast to trip")
    flagged = [a["code"] for a in status.get("alarms", [])] + [w["code"] for w in status.get("warnings", [])]
    codes = list(
        dict.fromkeys(flagged + sorted(entry, key=lambda c: abs(entry[c]["margin_pu"] / (entry[c]["limit_pu"] or 1))))
    )
    code = st.selectbox(
        "Alarm signal", codes, format_func=lambda c: f"{c} · {entry[c]['name']}", key=f"fc_{prof['name']}"
    )
    e = entry[code]
    k = e["scale"] or 1.0
    unit = eng(e, 0, t)[1]
    times = pd.to_datetime(pd.Series(fc["times"]))
    future = times.iloc[-1] + pd.to_timedelta(range(1, len(e["forecast_pu"]) + 1), unit="min")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=times, y=[v * k for v in e["history_pu"]], mode="lines", name="measured", line=dict(color=ACCENT, width=2)
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future,
            y=[v * k for v in e["forecast_pu"]],
            mode="lines",
            name="forecast",
            line=dict(color="#a78bfa", width=2, dash="dot"),
        )
    )
    fig.add_hline(
        y=e["limit_pu"] * k,
        line=dict(color=COLOURS["ALARM"], dash="dash", width=1),
        annotation_text="alarm limit",
        annotation_position="top left",
        annotation_font=dict(color="#ff8fa3"),
    )
    if e["minutes_to_trip"]:
        i = e["minutes_to_trip"] - 1
        fig.add_trace(
            go.Scatter(
                x=[future[i]],
                y=[e["forecast_pu"][i] * k],
                mode="markers+text",
                name="trip",
                text=[f"trip in {e['minutes_to_trip']} min"],
                textposition="top center",
                marker=dict(color=COLOURS["ALARM"], size=10),
                textfont=dict(color="#ff8fa3"),
            )
        )
    chart(
        style(fig, f"{code} · {e['signal']} ({unit}) · measured vs {fc['horizon']}-min forecast", 320), key="fc_chart"
    )


def reset_checklist(keys: list[str]) -> None:
    for k in keys:
        st.session_state[k] = False


def ask_llm(api: str, name: str, code: str, at: str, cache: dict) -> None:
    """Runs on a thread: an LLM answer takes longer than the auto-refresh, which would cancel it in-script."""

    def run():
        try:
            r = requests.get(f"{api}/advice", params={"profile": name, "code": code, "llm": True}, timeout=180)
            r.raise_for_status()
            ans = r.json()
            cache[(name, code)] = {
                "text": ans.get("llm_summary") or "(no answer)",
                "context": ans["context"],
                "time": at,
            }
        except requests.RequestException as exc:
            cache[(name, code)] = {"text": f"Advice request failed: {exc}", "context": "", "time": at}

    cache[(name, code)] = {"pending": True, "time": at}
    threading.Thread(target=run, daemon=True).start()


def view_advisor(api: str, prof: dict, status: dict, fc: dict, llm_on: bool) -> None:
    name, t = prof["name"], prof["asset_type"]
    entry = {a["code"]: a for a in fc["alarms"]}
    names = {a["code"]: a["name"] for a in prof["alarms"]}
    flagged = [a["code"] for a in status.get("alarms", [])] + [w["code"] for w in status.get("warnings", [])]
    codes = list(dict.fromkeys(flagged + list(names)))
    top = (status.get("advice") or {}).get("code")

    def label(c: str) -> str:
        return f"{ {'ALARM': '🔴', 'WARNING': '🟠'}.get(alarm_state(c, status), '🟢') } {c} · {names.get(c, '')}"

    c1, c2 = st.columns([3, 1])
    with c2:
        follow = st.toggle(
            "Follow live alarm",
            value=True,
            key="follow",
            help="Show advice for the highest-priority active or predicted alarm",
        )
    with c1:
        if follow and top:
            code = top
            st.selectbox("Alarm", [top], format_func=label, disabled=True, key="adv_follow")
        else:
            code = st.selectbox("Alarm", codes, format_func=label, key=f"adv_{name}")
            if follow:
                st.caption("No live alarm or prediction right now: pick any code to browse its manual entry.")

    kb = get_static(api, "/advice", profile=name, code=code)
    state = alarm_state(code, status)
    e = entry.get(code)
    live = []
    if e:
        live.append(
            f"Now <b>{' '.join(eng(e, e['value_pu'], t))}</b> · limit "
            f"{'>' if e['direction'] == 'above' else '<'} {' '.join(eng(e, e['limit_pu'], t))}"
        )
        if not e["condition_met"]:
            live.append(f"⚪ Idle: the alarm only applies when the condition '{e['condition']}' holds")
        elif e["minutes_to_trip"]:
            live.append(f"Trend forecast: trips in <b>~{e['minutes_to_trip']} min</b>")
        else:
            live.append("Trend forecast: no crossing in the next hour")
    st.markdown(alarm_card(code, kb["title"], live, COLOURS[state]), unsafe_allow_html=True)

    checks, steps = kb.get("checks", []), kb.get("steps", [])
    keys = [f"ck_{name}_{code}_c{i}" for i in range(len(checks))] + [
        f"ck_{name}_{code}_s{i}" for i in range(len(steps))
    ]
    done = sum(bool(st.session_state.get(k)) for k in keys)
    p1, p2 = st.columns([4, 1])
    p1.progress(done / max(len(keys), 1), text=f"Checklist {done}/{len(keys)} done")
    p2.button("↺ Reset", on_click=reset_checklist, args=(keys,), width="stretch", key=f"reset_{name}_{code}")

    left, right = st.columns(2)
    with left:
        st.markdown("##### 🔎 Likely causes")
        st.markdown("\n".join(f"- {c}" for c in kb.get("causes", [])))
        st.markdown("##### ✅ Inspection checkpoints")
        for i, c in enumerate(checks):
            st.checkbox(c, key=keys[i])
    with right:
        st.markdown("##### 🛠 Corrective steps")
        for i, s in enumerate(steps):
            st.checkbox(f"{i + 1}. {s}", key=keys[len(checks) + i])
    if kb.get("safety"):
        st.warning(f"**Safety:** {kb['safety']}", icon="⚠️")
    st.caption(f"Source: {kb.get('source')}")

    st.markdown("#### 🤖 AI maintenance recommendation")
    cache = st.session_state.setdefault("llm", {})
    a1, a2 = st.columns([1, 1])
    hit = cache.get((name, code), {})
    a1.button(
        "Generate from live readings",
        type="primary",
        key=f"ask_{name}_{code}",
        disabled=bool(hit.get("pending")),
        on_click=ask_llm,
        args=(api, name, code, status.get("time", ""), cache),
    )
    auto = a2.toggle("Auto-generate for each new live alarm", value=False, key="auto_llm")
    if not llm_on:
        st.caption(
            "The API has no OLLAMA_URL set, so requests go to a local Ollama at http://127.0.0.1:11434 "
            "(model OLLAMA_MODEL, default llama3.2). The checklist above works without an LLM."
        )
    if auto and follow and code == top and not hit:
        ask_llm(api, name, code, status.get("time", ""), cache)
        hit = cache[(name, code)]
    if hit.get("pending"):
        st.info("⏳ The LLM is writing a recommendation from the live readings; it appears here when ready.")
    elif hit:
        with st.container(border=True):
            st.caption(f"Generated at telemetry time {hit['time']}")
            st.markdown(hit["text"])
            with st.expander("Live context sent to the LLM"):
                st.code(hit["context"], language=None, wrap_lines=True)


def view_profile(prof: dict) -> None:
    t = prof["asset_type"]
    left, right = st.columns([1, 2])
    with left:
        st.markdown("#### Rating")
        st.dataframe(
            pd.DataFrame({"Parameter": list(prof["rating"]), "Value": [str(v) for v in prof["rating"].values()]}),
            hide_index=True,
            width="stretch",
        )
    with right:
        st.markdown("#### Signal map")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Raw column": col,
                        "Role": role,
                        "Label": SIGNALS[t].get(role, (role,))[0],
                        "Unit": SIGNALS[t].get(role, ("", "", ""))[1],
                        "Group": SIGNALS[t].get(role, ("", "", "–"))[2],
                    }
                    for col, role in prof["signals"].items()
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    st.markdown("#### Alarm configuration")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Code": a["code"],
                    "Alarm": a["name"],
                    "Signal (pu)": a["signal"],
                    "Trips when": f"{a['direction']} {a['limit_pu']:g} pu",
                    "Limit (raw)": ", ".join(f"{x:.4g}" for x in a["raw_limits"]) or "–",
                    "Condition": a["condition"],
                    "Warning horizon (min)": a["warn_minutes"],
                }
                for a in prof["alarms"]
            ]
        ),
        hide_index=True,
        width="stretch",
    )


# ------------------------------------------------------------------ layout
st.markdown(CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Control panel")
    api = st.text_input("API URL", "http://127.0.0.1:8000").rstrip("/")
    try:
        health = get(api, "/health")
        profs = get_static(api, "/profiles")
    except requests.RequestException as exc:
        st.error("API not reachable")
        st.markdown(
            f'<div class="empty"><b>Cannot reach the API at {api}</b><br><br>Start it from the repo root:<br>'
            "<code>uvicorn pdm.serve.api:app --port 8000</code><br><br>"
            f"<small>{type(exc).__name__}</small></div>",
            unsafe_allow_html=True,
        )
        st.stop()

    by_name = {p["name"]: p for p in profs}
    default = by_name.get(health["profile"], profs[0])
    asset_type = (
        st.segmented_control(
            "Asset type",
            list(ASSET_TYPES),
            format_func=ASSET_TYPES.get,
            default=default["asset_type"],
            key="asset_type",
        )
        or default["asset_type"]
    )
    names = sorted(
        (p["name"] for p in profs if p["asset_type"] == asset_type), key=lambda n: float(rating_label(n).split()[0])
    )
    name = st.selectbox(
        "Power rating",
        names,
        format_func=rating_label,
        key=f"rating_{asset_type}",
        index=names.index(default["name"]) if default["name"] in names else 0,
    )
    prof = by_name[name]

    st.markdown("### 📡 Live feed")
    st.caption("Streams digital-twin telemetry for the selected asset from inside the API.")
    scenario = st.radio(
        "Scenario",
        ["fault", "normal"],
        horizontal=True,
        format_func={"fault": "Fault demo", "normal": "Normal day"}.get,
    )
    speed = st.slider("Speed (telemetry minutes per second)", 1, 30, 2)
    feed = health["assets"].get(name, {}).get("feed", {})
    b1, b2 = st.columns(2)
    if b1.button("▶ Start", type="primary", width="stretch"):
        with st.spinner("Generating twin telemetry…"):
            post(api, "/feed/start", profile=name, scenario=scenario, speed=speed)
        st.toast(f"Live feed started for {name}", icon="📡")
    if b2.button("⏹ Stop", width="stretch", disabled=not feed.get("running")):
        post(api, "/feed/stop", profile=name)
        st.toast("Live feed stopped", icon="⏹")

    st.markdown("### 🖥 Display")
    auto = st.toggle("Live auto-refresh", value=True)
    refresh = st.slider("Refresh every (s)", 1, 10, 2)
    window = st.select_slider(
        "History window", [60, 120, 240, 480, 720, 1440], value=240, format_func=lambda m: f"{m // 60} h"
    )
    st.caption(
        "LLM advisor: "
        + ("🟢 Ollama configured on the API" if health.get("llm") else "⚪ manual knowledge base (LLM on request)")
    )


@st.fragment(run_every=refresh if auto else None)
def live() -> None:
    try:
        status = get(api, "/status", profile=name)
        hist = pd.DataFrame(get(api, "/history", profile=name, n=window))
        fc = get(api, "/forecast", profile=name)
    except requests.RequestException as exc:
        st.error(f"API request failed: {exc}")
        return
    hero(prof, status)
    if hist.empty:
        st.markdown(
            f'<div class="empty"><b>No telemetry yet for {name}</b><br><br>'
            "Press <b>▶ Start</b> under <i>Live feed</i> in the sidebar to stream twin data, or post rows with<br>"
            f"<code>python scripts/replay.py --profile {name} --mode api --speed 50</code></div>",
            unsafe_allow_html=True,
        )
        return
    hist = hist.assign(timestamp=pd.to_datetime(hist["timestamp"])).set_index("timestamp")
    if status.get("status") == "warming_up":
        st.progress(
            min(status["rows"] / 60, 1.0),
            text=f"Warming up: {status['rows']}/60 minutes of history before the agents start",
        )
    kpi_cards(prof, hist, status)
    view = st.segmented_control("View", VIEWS, default=VIEWS[0], key="view", label_visibility="collapsed") or VIEWS[0]
    if view == VIEWS[0]:
        view_telemetry(prof, hist, status)
    elif view == VIEWS[3]:
        view_profile(prof)
    elif not fc["alarms"] or status.get("status") == "warming_up":
        st.info("Alarms, forecasts and advice start once 60 minutes of history are in.")
    elif view == VIEWS[1]:
        view_alarms(prof, status, fc)
    else:
        view_advisor(api, prof, status, fc, bool(health.get("llm")))


live()
