# Real data (not included)

The cloud environment used to build this project could not download datasets, so real files are not bundled.
Download them yourself and place CSVs here.

| Dataset | Use | Link |
| --- | --- | --- |
| La Reunion rooftop PV (2 inverters, 5 s, 12+ months) | Real weather and inverter base telemetry | Zenodo DOI 10.5281/zenodo.7157424 |
| NASA Li-ion battery aging | Real battery voltage/temperature behaviour | https://data.nasa.gov/dataset/li-ion-battery-aging-datasets |
| GPVS-Faults (PV inverter lab faults) | Real-fault check with scripts/validate_real.py | https://data.mendeley.com/datasets/n76t439f65/1 |
| Inverter-driven PMSM faults (10 Hz) | Real-fault check with scripts/validate_real.py | https://zenodo.org/records/14482932 |

To use real weather in the twin: build a DataFrame with columns `g` (W/m2), `t_amb` (degC), `hour`
at 1-minute steps and pass it as `generate(..., weather=df)`.

For `scripts/validate_real.py`, reshape the dataset to: `timestamp, run_id, label, <signals...>`.
For GPVS-Faults, first reduce the ~100 kHz signals to short-window features (RMS, mean) at about 10 Hz.
