# Agentic AI Multi-Agent Decision Support Platform for Industrial IoT Predictive Maintenance

**M.Tech Phase II project, PES University** · Sumanth GM (PES2PG24DS312) · Guide: Dr. Pulkit Dwivedi

The platform warns **30–120 minutes before** a hybrid solar inverter or a solar pump drive trips, works on **any power
rating without retraining** (one YAML profile per device), and returns a **debug checklist** for the predicted alarm.
It extends the ASPIRE baseline (Datta & Guha, IEEE Access 2026), reproduced in Phase I.

| | |
| --- | --- |
| Assets | Hybrid inverter (PV + battery + grid + generator), solar water-pump drive |
| Alarms predicted | 19 (11 inverter, 8 pump drive) |
| Power ratings | 3, 5, 10 kW and 3, 5, 10 HP, one model per asset type |
| Model | XGBoost, one classifier per alarm, on per-unit window features |
| Real data | PMSM inverter-drive lab faults (E5), La Reunion measured PV weather (E6) |
| Runs in | VS Code (Windows / macOS / Linux) and Google Colab, CPU only |

---

## Contents

1. [How it works](#how-it-works)
2. [What is real and what is simulated](#what-is-real-and-what-is-simulated)
3. [Results](#results)
4. [Repository structure](#repository-structure)
5. [Quick start (VS Code)](#quick-start-vs-code)
6. [Reproduce every result](#reproduce-every-result)
7. [Google Colab](#google-colab)
8. [Live demo](#live-demo)
9. [Add a new asset or rating](#add-a-new-asset-or-rating)
10. [Limitations](#limitations)

More detail: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · [`docs/RESULTS.md`](docs/RESULTS.md) · [`data/real/README.md`](data/real/README.md)

---

## How it works

Every 5 minutes the platform reads the last 60 minutes of a device's telemetry and runs four agents.

```
 asset profile (YAML)        telemetry (twin, real weather, or live HTTP)
          \                          /
           +--> per-unit conversion + 60-min window features
                          |
     +-----------+--------+---------+-------------+
     |           |                  |             |
  Monitor   Early-Warning        Forecast      Advisor
  (rules)   (XGBoost/alarm)    (trend line)  (knowledge base)
     +-----------+--------+---------+-------------+
                          |
          Orchestrator: NORMAL / WARNING / ALARM
          + minutes to trip + fix checklist
                          |
             FastAPI service -> Streamlit dashboard
```

| Agent | Type | Output |
| --- | --- | --- |
| Monitor | Rules from the profile | Alarms whose limit is crossed now |
| Early-Warning | **Trained model** (XGBoost, one per alarm) | Probability that each alarm trips within its horizon |
| Forecast | Least-squares trend of the last 30 min | Minutes until the warned signal crosses its limit |
| Advisor | Lookup in `knowledge/fault_kb.yaml` (optional local LLM wording) | Causes, checks, steps, safety note, source |

**Main idea (novelty):** every signal is divided by its rating or trip limit from the profile (per-unit), and two
limit-aware features are added (distance to limit, time to limit). A model trained on a 5 kW unit then works on 3 kW
and 10 kW units by changing only the profile. Full description: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## What is real and what is simulated

| Experiment | Device telemetry | Faults | Weather | Folder |
| --- | --- | --- | --- | --- |
| E1–E4 | Simulated (telemetry twin) | Simulated, injected | Simulated | `results/synthetic_weather/` |
| **E5** | **Real** (PMSM lab drive, 10 Hz) | **Real** (9 classes) | – | `results/real_faults/` |
| **E6** | Simulated (twin) | Simulated, injected | **Real** (La Reunion, 519 days) | `results/real_weather/` |

No public dataset labels the 19 inverter and drive alarms, so the twin generates labelled telemetry with true fault
episodes and near-misses. E5 checks the same features and XGBoost on real faults (a different label set, re-trained);
E6 drives the twin with measured weather. Dataset sources and licences: [`data/real/README.md`](data/real/README.md).

## Results

Full tables: [`docs/RESULTS.md`](docs/RESULTS.md). Train on 5 kW / 5 HP; test on 40 unseen days × 3 seeds per rating.

**E1–E2 · Early warning and unseen ratings (macro-F1, simulated weather)**

| Asset | Method | 5 (trained) | 3 (unseen) | 10 (unseen) | False alerts/day |
| --- | --- | --- | --- | --- | --- |
| Inverter (kW) | **Ours (per-unit XGBoost)** | **0.645** | **0.641** | **0.643** | 2.7 |
| | Raw-feature XGBoost (ablation) | 0.588 | 0.432 | 0.568 | 2.0 |
| | ASPIRE-style Transformer (raw) | 0.385 | 0.021 | 0.286 | 4.9 |
| Pump drive (HP) | **Ours (per-unit XGBoost)** | **0.572** | **0.591** | **0.575** | 0.8 |
| | Raw-feature XGBoost (ablation) | 0.500 | 0.396 | 0.359 | 1.1 |
| | ASPIRE-style Transformer (raw) | 0.215 | 0.167 | 0.183 | 9.0 |

**E3 · Forecast:** the trend method caught 10 of 12 thermal trips 30 min ahead (mean error 4.6 / 12.4 min); persistence and seasonal caught none.
**E4 · Advisor:** correct entry ranked first for 95.6% of 997 alarm events, against 10.1% for README-style RAG on raw values.

**E5 · Real faults (PMSM, time-blocked 4-fold, electrical signals only, F0–F5)**

| Features | Macro-F1 | Accuracy | Fault-detection F1 |
| --- | --- | --- | --- |
| **20-sample (2 s) window + XGBoost** | **0.960** | 0.966 | 0.968 |
| Latest sample only | 0.632 | 0.745 | 0.834 |

**E6 · Measured weather (macro-F1, trained → unseen 3 / 10)**

| Asset | Ours (per-unit) | Raw-feature ablation |
| --- | --- | --- |
| Inverter | 0.579 → 0.577 / 0.570 | 0.584 → 0.429 / 0.515 |
| Pump drive | 0.566 → 0.546 / 0.570 | 0.512 → 0.316 / 0.389 |

## Repository structure

```
pdm_agentic/
├── README.md                    this file
├── .github/workflows/ci.yml     tests + quick experiment run on every push
├── .vscode/                     Run and Debug configurations, settings
├── pyproject.toml               package definition (pip install -e .)
├── requirements.txt             all dependencies (core, baseline, serving, dev)
├── configs/
│   ├── experiment.yaml          days, seeds, episodes, methods (full and --quick)
│   └── profiles/                6 asset profiles: inverter 3/5/10 kW, pump drive 3/5/10 HP
├── knowledge/fault_kb.yaml      19 alarm entries: causes, checks, steps, safety, source
├── src/pdm/                     the package
│   ├── profiles.py              profile loading, per-unit transform, distance to limit
│   ├── twin/                    telemetry twin: weather (synthetic or measured), inverter, drive, faults
│   ├── labels.py                Normal / Warning / Alarm labels (ASPIRE-style)
│   ├── features.py              60-minute window features
│   ├── agents/                  monitor, early_warning, forecast, advisor, orchestrator
│   ├── baseline/                ASPIRE-style Transformer (comparison only)
│   ├── eval/                    metrics, figures, report writer
│   ├── experiments.py           E1–E4 runner
│   ├── training.py              train and save the live models
│   └── serve/api.py             FastAPI service
├── scripts/
│   ├── run_experiments.py       E1–E4 (and E6 with --weather)
│   ├── train.py                 train the live models -> results/models/
│   ├── prepare_pmsm.py          download + convert the PMSM real-fault dataset (E5)
│   ├── prepare_reunion_weather.py  download + convert La Reunion weather (E6)
│   ├── validate_real.py         E5 real-fault check
│   └── replay.py                stream twin data to the API or console
├── app/dashboard.py             Streamlit dashboard
├── notebooks/Phase2_validation.ipynb   Colab / VS Code notebook (all experiments)
├── data/real/                   real datasets (CSV files are created by the prepare scripts)
├── results/
│   ├── models/                  saved XGBoost models used by the live API
│   ├── synthetic_weather/       E1–E4 tables, figures, report.md
│   ├── real_faults/             E5 JSON summaries and confusion matrices
│   └── real_weather/            E6 tables, figures, report.md
├── docs/
│   ├── ARCHITECTURE.md          system design, data flow, design decisions
│   └── RESULTS.md               all experiment results with explanations
└── tests/                       pytest: core, API, real-data loaders and splits
```

## Quick start (VS Code)

Open the repository folder (`pdm_agentic`, the one that contains `src/`) in VS Code, then in the terminal:

```bash
python -m venv .venv
.venv\Scripts\activate              # macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest -q                           # 9 tests
python scripts/run_experiments.py --quick    # about 2 minutes, writes results_quick/
```

Every step also has a ready configuration in **Run and Debug** (`.vscode/launch.json`).

## Reproduce every result

```bash
# E1–E4 (simulated weather), about 30 min on a laptop CPU
python scripts/run_experiments.py                      # -> results/synthetic_weather/
python scripts/train.py --profile inverter_hybrid_5kw  # live models -> results/models/
python scripts/train.py --profile pump_drive_5hp

# E5 real faults (1 MB download)
python scripts/prepare_pmsm.py
python scripts/validate_real.py --csv data/real/pmsm_10hz.csv --split blocks --window 20 \
    --signals Ia Ib VDC IDC VD --labels F0 F1 F2 F3 F4 F5 --out results/real_faults --tag electrical_window
python scripts/validate_real.py --csv data/real/pmsm_10hz.csv --split blocks --window 20 --features instant \
    --signals Ia Ib VDC IDC VD --labels F0 F1 F2 F3 F4 F5 --out results/real_faults --tag electrical_instant

# E6 measured weather (streams a 2.2 GB file, about 6 min)
python scripts/prepare_reunion_weather.py
python scripts/run_experiments.py --weather data/real/reunion_weather_1min.csv --out results/real_weather
```

The ASPIRE-style Transformer needs PyTorch (`pip install torch`); without it, that baseline is skipped.

## Google Colab

Open `notebooks/Phase2_validation.ipynb` in Colab and run all cells. The first cell clones the repository (set
`GIT_URL`) or asks you to upload `pdm_agentic.zip`. Switches at the top choose between showing the saved results
(fast) and re-running each experiment (`RUN_E1_E4`, `RUN_E5`, `RUN_E6`).

## Live demo

```bash
uvicorn pdm.serve.api:app --port 8000      # terminal 1
streamlit run app/dashboard.py             # terminal 2
```

In the dashboard pick the asset type and rating and press **Start** under *Live feed*. *Fault demo* starts 45 minutes
before an alarm, so the warning, forecast and alarm appear within a minute. Tabs: live telemetry, alarms and forecast,
maintenance advisor (optional Ollama LLM: set `OLLAMA_URL`, `OLLAMA_MODEL`), asset profile.
Set `PDM_WEATHER=data/real/reunion_weather_1min.csv` before starting the API to drive the live feed with measured weather.

API: `GET /health`, `/profiles`, `/status`, `/history`, `/forecast`, `/advice?code=&llm=`, `POST /ingest`,
`/feed/start?scenario=fault|normal&speed=`, `/feed/stop`. Every endpoint takes `?profile=<name>`.
Without the API: `python scripts/replay.py --profile pump_drive_3hp --mode local`.

## Add a new asset or rating

1. Copy a profile in `configs/profiles/`, change the `rating` block to the new datasheet values and map raw signal names in `signals`.
2. Set each alarm's `limit_pu` and `warn_minutes` from the manual.
3. Run `python scripts/replay.py --profile <new_profile> --mode local`. No retraining is needed within an asset type.
4. For a new asset type, add its per-unit rules in `Profile.to_pu`, a twin generator, its alarms, and knowledge-base entries.

## Limitations

- The 19 alarm faults are injected into simulated telemetry; field logs from a deployed inverter or drive are the next step.
- E5 uses real faults but a different label set (component faults, not the 19 alarms) and tests classification, not early warning; each class is a single recording, so the electrical-only result is the one to quote.
- E6 uses real weather but simulated device behaviour, and the Transformer baseline was not re-run there.
- Ratings and limits in the profiles are realistic placeholders; replace them with datasheet values for a specific device.
- The knowledge base is written from general practice; replace each `source` with the cited manual page.
- The ASPIRE-style baseline re-implements the Prediction Agent's Transformer only, not the full ASPIRE system.

## References

- A. Datta and S. Guha, "ASPIRE: An agentic decision system for early equipment failure prediction and intervention in industrial IIoT," *IEEE Access*, vol. 14, 2026, doi: 10.1109/ACCESS.2026.3711010.
- Inverter-driven PMSM fault dataset, Zenodo, doi: 10.5281/zenodo.14482932 (CC-BY-4.0).
- La Reunion rooftop PV and meteorological dataset, Zenodo, doi: 10.5281/zenodo.7157424 (CC-BY-4.0).
