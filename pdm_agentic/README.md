# Agentic AI Multi-Agent Decision Support Platform for Industrial IoT Predictive Maintenance

M.Tech Phase II framework. It predicts the alarms that trip **hybrid inverters** and **solar pump drives** before they happen,
works across **power ratings and asset types by swapping one profile file**, and gives a **cited debug checklist** for each alarm.
Baseline: ASPIRE (Datta & Guha, IEEE Access 2026). Reference pipeline: the PLC Predictive Maintenance + RAG project.

## What is new compared with the baseline

| Item | ASPIRE (as reproduced in Phase I) | This framework |
| --- | --- | --- |
| Assets | Metro air compressor (MetroPT-3) | Hybrid inverter, solar pump drive |
| Alarm types predicted | 1 failure category | 19 (11 inverter + 8 drive) |
| Power ratings | 1 | 3 per asset type, one model, profile swap |
| Fix guidance | None | Cited checklist per alarm code |

Three claims, each with an experiment in `scripts/run_experiments.py`:

1. **Multi-fault early warning** (E1) - per-alarm F1, event recall, lead time, false alerts per day.
2. **Rating- and asset-agnostic** (E2) - train on 5 kW / 5 HP, test on 3 kW, 10 kW, 3 HP, 10 HP with no retraining.
3. **Alarm to action** (E4) - Advisor returns the right manual entry, compared with README-style RAG.
Plus the Forecast Agent's time-to-trip (E3).

## Results (full run, `results/report.md`)

Telemetry twin, 40 test days per unit, 3 seeds per rating. Macro-F1 of early warning (higher is better):

| Asset | Method | 5 kW / 5 HP (trained) | 3 kW / 3 HP | 10 kW / 10 HP | False alerts/day (trained) |
| --- | --- | --- | --- | --- | --- |
| Inverter | **Ours (per-unit XGBoost)** | **0.645** | **0.641** | **0.643** | 2.7 |
| Inverter | Raw-feature XGBoost (ablation) | 0.588 | 0.432 | 0.568 | 2.0 |
| Inverter | ASPIRE-style Transformer (raw) | 0.385 | 0.021 | 0.286 | 4.9 |
| Inverter | ASPIRE-style Transformer (per-unit) | 0.425 | 0.413 | 0.428 | 5.4 |
| Pump drive | **Ours (per-unit XGBoost)** | **0.572** | **0.591** | **0.575** | 0.8 |
| Pump drive | Raw-feature XGBoost (ablation) | 0.500 | 0.396 | 0.359 | 1.1 |
| Pump drive | ASPIRE-style Transformer (raw) | 0.215 | 0.167 | 0.183 | 9.0 |
| Pump drive | ASPIRE-style Transformer (per-unit) | 0.332 | 0.325 | 0.319 | 5.2 |

- Advisor (E4): correct knowledge-base entry first for 98.8% of inverter and 91.0% of pump-drive alarm events,
  against 9.2% and 11.4% for README-style RAG on raw sensor values.
- Forecast (E3): a 30-minute-ahead trend forecast caught 5 of 6 thermal trips per asset type; persistence and seasonal naive caught none.

## Architecture

```
            asset profile (YAML: ratings, signal roles, limits, horizons)
                               |
telemetry -> MQTT/HTTP -> FastAPI -> buffer/InfluxDB -> per-unit features
                                                         |
                    +------------------+-----------------+------------------+
                    |                  |                                    |
              Monitor Agent     Early-Warning Agent                 Forecast Agent
              (live alarms)     (XGBoost per alarm, SHAP-ready)     (trend / Chronos: time to trip)
                    +------------------+-----------------+------------------+
                                                         |
                                                   Advisor Agent
                                    (alarm code -> knowledge base -> cited checklist; optional local LLM)
                                                         |
                                                 Streamlit dashboard
```

## Folder structure

```
configs/profiles/      6 asset profiles (3, 5, 10 kW inverters; 3, 5, 10 HP pump drives)
configs/experiment.yaml  experiment settings (full and --quick)
knowledge/fault_kb.yaml  fault knowledge base for the Advisor (replace sources with manual pages)
src/pdm/profiles.py    profile loading and the per-unit transform
src/pdm/twin/          telemetry twin: weather, healthy inverter/drive, fault injection
src/pdm/labels.py      ASPIRE-style labels: Normal / Warning / Alarm
src/pdm/features.py    per-unit window features, raw features, sequences
src/pdm/agents/        monitor, early_warning, forecast, advisor, orchestrator
src/pdm/baseline/      ASPIRE-style Transformer (d_model 64, 4 heads, 2 layers, FF 128, dropout 0.2)
src/pdm/eval/          metrics and report generator
src/pdm/serve/api.py   FastAPI service (/ingest, /status, /history)
app/dashboard.py       Streamlit dashboard
scripts/               run_experiments, train, replay, validate_real
notebooks/Phase2_validation.ipynb   Colab / VS Code notebook that reproduces the results
results/               report.md, CSV tables, figures, trained models
tests/                 pytest smoke tests
```

## Run it in VS Code

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pytest -q
python scripts/run_experiments.py --quick               # about 1-2 minutes
python scripts/run_experiments.py                       # full results -> results/report.md
```

The `.vscode/launch.json` file has ready-made run configurations: experiments, training, API, replay and dashboard.

Live demo (two terminals, or the launch configurations):

```bash
uvicorn pdm.serve.api:app --port 8000
streamlit run app/dashboard.py
```

In the dashboard sidebar pick the asset type (hybrid inverter / solar pump drive) and power rating, then press
**Start** under *Live feed*. The API streams twin telemetry for that asset; *Fault demo* starts 45 minutes before an
alarm so the early warning, forecast and alarm appear within a minute. The views are:

- **Live telemetry**: KPI cards with sparklines and alarm state, plus a chart for each signal with its alarm limits.
- **Alarms & forecast**: active and predicted alarms, a watchlist of every alarm, and the forecast-to-trip chart.
- **Maintenance advisor**: the manual's causes, checkpoints and steps as a checklist, and an LLM recommendation
  written from the live readings. It uses Ollama: set `OLLAMA_URL` (and optionally `OLLAMA_MODEL`, default `llama3.2`)
  before starting the API, or run Ollama locally on port 11434.
- **Asset profile**: rating, signal map and alarm configuration.

To feed your own rows instead, post them to the API (every endpoint takes `?profile=<name>`):

```bash
python scripts/replay.py --profile inverter_hybrid_10kw --mode api --speed 50
```

API endpoints: `GET /health`, `/profiles`, `/status`, `/history`, `/forecast`, `/advice?code=&llm=`,
`POST /ingest`, `/feed/start?scenario=fault|normal&speed=`, `/feed/stop`.

Or without the API: `python scripts/replay.py --profile pump_drive_3hp --mode local`.

## Run it in Google Colab

Upload `notebooks/Phase2_validation.ipynb` to Colab, run the first cell and upload `pdm_agentic.zip` when asked
(or set `GIT_URL` to your repository). The notebook runs all experiments, shows every table and figure, and runs
the live-style demo.

## Add a new asset or power rating

1. Copy a profile in `configs/profiles/`, change the `rating` values to the new datasheet, and map raw signal names in `signals`.
2. Set each alarm's `limit_pu` and `warn_minutes` from the manual.
3. Run `python scripts/replay.py --profile <new_profile> --mode local`. No retraining is needed within the same asset type.
4. For a new asset type, add its per-unit rules in `Profile.to_pu`, a twin generator, alarms in the profile, and entries in `knowledge/fault_kb.yaml`.

## Data

All experiments run on the telemetry twin: healthy telemetry generated from weather, with injected faults that grow
gradually (true episodes cross the limit; near-miss episodes stop short). Real datasets for base telemetry and real-fault
checks are listed in `data/real/README.md`; use `scripts/validate_real.py` on them.

## Limitations (state these in the report)

- Faults are injected; thresholds are placeholders until copied from the reference device's datasheet.
- Per-unit features remove rating scale by design in the twin; confirm the cross-rating result on real multi-rating data.
- The knowledge base was written from general practice; replace each `source` with the cited manual page.
- The ASPIRE-style baseline re-implements the Prediction Agent's Transformer, not the full ASPIRE system.
