# Real data

Raw files are not kept in git (`.gitignore`); the scripts below download and convert them.

| Dataset | Use | Script | Output |
| --- | --- | --- | --- |
| La Reunion rooftop PV, 4 kW, 1 s, Oct 2021 - Apr 2023 (Zenodo 10.5281/zenodo.7157424, CC-BY-4.0) | Measured irradiance and ambient temperature as the twin's weather | `python scripts/prepare_reunion_weather.py` (streams the 2.2 GB file, about 6 min) | `reunion_weather_1min.csv` (25 MB, 519 complete days) |
| Inverter-driven PMSM faults, 10 Hz (Zenodo 10.5281/zenodo.14482932, CC-BY-4.0) | Real-fault check of the window features + XGBoost | `python scripts/prepare_pmsm.py` (1 MB zip) | `pmsm_10hz.csv` (9 classes, one recording each) |
| GPVS-Faults (Mendeley n76t439f65) | Optional second real-fault check | not scripted yet: reduce the ~100 kHz signals to RMS/mean features at about 10 Hz | `timestamp, run_id, label, <signals>` |
| NASA Li-ion battery aging | Optional battery behaviour | not used | |

## Using the measured weather

```bash
python scripts/run_experiments.py --weather data/real/reunion_weather_1min.csv --out results/real_weather
python scripts/train.py --profile inverter_hybrid_5kw --weather data/real/reunion_weather_1min.csv
PDM_WEATHER=data/real/reunion_weather_1min.csv uvicorn pdm.serve.api:app --port 8000   # live feed
```

`GTI` (irradiance on the panel plane, W/m2) becomes `g`; `TA` (ambient, degC) becomes `t_amb`. Timestamps are local
time (UTC+4). `pdm.twin.load_weather` fills gaps up to 30 min and drops days that still have missing minutes. In the
experiments, training uses days 0-59 and test seed *i* uses its own later 40-day block, so test weather is never seen
in training. Device telemetry and faults are still produced by the twin.

## Real-fault check (PMSM)

```bash
python scripts/validate_real.py --csv data/real/pmsm_10hz.csv --split blocks --window 20 --out results/real_faults
python scripts/validate_real.py --csv data/real/pmsm_10hz.csv --split blocks --window 20 --features instant --out results/real_faults --tag instant
python scripts/validate_real.py --csv data/real/pmsm_10hz.csv --split blocks --window 20 \
    --signals Ia Ib VDC IDC VD --labels F0 F1 F2 F3 F4 F5 --out results/real_faults --tag electrical_window
```

Each class is one recording, so leave-one-run-out is impossible. `--split blocks` cuts every recording into 4 time
blocks; each fold tests one block and trains only on windows that share no samples with it. The temperature channels
differ between recordings even without an overheating fault, so a model can recognise the session instead of the
fault; the electrical-only run on the switch faults (F0-F5) is the fairer test.
