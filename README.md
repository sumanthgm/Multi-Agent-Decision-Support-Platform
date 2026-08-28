#Agentic Decision System for Early Equipment Failure Prediction

Reproduction scaffold for:

> S. Guha and A. Datta, "ASPIRE: An Agentic Decision System for Early Equipment
> Failure Prediction and Intervention in Industrial IIoT," **IEEE Access**, vol. 14,
> pp. 105006–105031, 2026. DOI: 10.1109/ACCESS.2026.3711010 (Open Access, CC BY 4.0)

This is a from-scratch, best-effort re-implementation based on the architecture,
equations, and hyperparameters **stated explicitly in the paper**. The authors did
not publish a code repository at the time of writing — search
`https://github.com/search?q=ASPIRE+IIoT+MetroPT` and the authors' own listed work
(`GDRNet`, IEEE Access 2025) periodically in case an official repo appears.

⚠️ **Honesty note on numbers**: The metrics quoted throughout `docs/` and in code
comments as "PAPER-REPORTED" are copied directly from the paper's tables and are
**not** independently verified here. Any run of this code produces "REPRODUCED
(THIS RUN)" numbers that will differ — Table 3's calibration was done on the
authors' own pre-holdout split, exact preprocessing, and random seeds, none of
which are shared beyond what's described in prose. Treat this repo as a faithful
*architecture* reproduction, not a guaranteed *score* reproduction.

## 1. Dataset

**MetroPT-3** (UCI Machine Learning Repository), Davari et al. 2021.
DOI: `10.24432/C5VW3R`
- UCI page: https://archive.ics.uci.edu/dataset/791/metropt+3+dataset
- Also mirrored on Kaggle: search "MetroPT-3 air compressor"

```bash
# From the UCI page (requires ucimlrepo package) — no login required
pip install ucimlrepo --break-system-packages
python -c "
from ucimlrepo import fetch_ucirepo
ds = fetch_ucirepo(id=791)
ds.data.features.to_csv('data/metropt3_raw.csv', index=False)
"
```

If `ucimlrepo` cannot resolve the dataset id, download the CSV manually from the
UCI page above and place it at `data/metropt3_raw.csv`. Columns expected (paper,
Section III-B): `TP2, TP3, H1, DV_pressure, Reservoirs, Oil_temperature,
Motor_current, COMP, DV_electric, Towers, MPG, LPS` plus `Pressure_switch,
Caudal_impulses, Oil_level` (dropped — see preprocessing) and a `timestamp` column.

## 2. Environment

```bash
pip install -r requirements.txt --break-system-packages
```

Paper's actual stack: Python 3, Keras/TensorFlow (neural models), NumPy/SciPy/
scikit-learn (numerical + JS-divergence/KS-test), SQLite (`sqlite3`, built-in) for
the EventStore, OpenAI Responses API (`gpt-4o-mini`) for the Expert Agent. Trained
on Google Colab (NVIDIA L4 GPU) — not required for reproduction, just faster.

## 3. Pipeline (run in order)

```bash
python src/data_pipeline.py          # clean, select sensors, window, label, split
python src/train_sensor_agents.py    # 12x per-sensor LSTM autoencoders
python src/train_adaptive_window.py  # GDRNet-style MLP + LSTM next-step predictor
python src/train_prediction_agent.py # Transformer 3-class classifier (the baseline)
python src/calibrate_decision_agent.py  # PR sweep -> Table 3 policy parameters
python src/evaluate.py               # Table 5/6/7/8 equivalents + ablation (Table 11)
```

Or end-to-end:
```bash
python src/pipeline.py --holdout_windows 2000
```

## 4. What each file implements (paper section → file)

| Paper section | File |
|---|---|
| III-C Data preprocessing & label construction | `src/data_pipeline.py` |
| II-B-1 Sensor Agent (Eq. 2–5) | `src/sensor_agent.py` |
| II-B-2 Master Sensor Aggregator | `src/master_aggregator.py` |
| II-B-3 Adaptive Window Agent (Eq. 6–12, GDRNet) | `src/adaptive_window_agent.py` |
| II-B-4 Prediction Agent (Transformer, Eq. 13) | `src/prediction_agent.py` |
| II-B-5 Decision Agent (Eq. 14–18, Algorithms 1 & 2, Table 3) | `src/decision_agent.py` |
| II-B-6 Expert Agent (Eq. 19) | `src/expert_agent.py` |
| III-F / IV-H Cost framework (Eq. 20–28) | `src/cost_framework.py` |
| IV-E Ablation study (Table 11, variants V0–V6) | `src/ablation.py` |
| III-E Evaluation protocol, agreement buckets (Table 7) | `src/evaluate.py` |

## 5. Reference implementations / open-source building blocks

These are **general-purpose open-source references** for the techniques the paper
uses — not an official ASPIRE repo (none was found):

- LSTM autoencoder for anomaly detection (pattern used by Sensor Agents):
  https://github.com/curiousily/Getting-Things-Done-with-Pytorch (LSTM-AE chapter),
  or Keras example: https://keras.io/examples/timeseries/timeseries_anomaly_detection/
- Transformer time-series classifier (pattern used by Prediction Agent):
  https://keras.io/examples/timeseries/timeseries_transformer_classification/
- Jensen–Shannon divergence: `scipy.spatial.distance.jensenshannon`
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html
- Two-sample Kolmogorov–Smirnov fallback: `scipy.stats.ks_2samp`
- MetroPT-3 official dataset repo / paper (Davari et al. 2021):
  https://github.com/adthina/metroPT (community mirrors vary — verify against the
  UCI DOI above, since no single canonical GitHub repo is authoritative)
- Related prior MetroPT studies cited by the paper (for cross-checking
  preprocessing choices): Barros et al. 2020, Davari et al. 2021 DSAA,
  Najjar et al. 2023 — see `docs/ASPIRE_Reproduction_Plan.docx` References section.
- OpenAI Python SDK (Expert Agent LLM calls): https://github.com/openai/openai-python
- If you prefer an agent-orchestration framework instead of the plain-Python
  agent classes used here (which is what the paper itself uses — it does **not**
  use LangGraph/CrewAI), optional alternatives are:
  https://github.com/langchain-ai/langgraph and https://github.com/crewAIInc/crewAI

## 6. Known gaps you must fill in to hit the paper's exact numbers

The paper omits some values needed for bit-exact reproduction (this is normal for
a methods paper, not a defect): the exact α/β policy-parameter values in Eq. 14–18
beyond what Table 3 gives, the exact GDRNet target-window-size ground truth
construction, and exact train/test/holdout window counts. `configs/policy_params.yaml`
encodes every numeric value Table 3 *does* give; anything else is marked
`# CALIBRATE` and must be swept on your own pre-holdout split per Section III-A/IV-F.
