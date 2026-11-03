# Phase II results: rating-aware multi-agent fault prediction

Test set: 40 days per unit, 3 random seed(s) per rating, unseen weather and fault timings. Training: one 60-day unit per asset type (5 kW inverter, 5 HP pump drive). All data is from the telemetry twin (see Limitations).

## E1 - Multi-fault early warning (trained rating, unseen seeds)

All methods see the same windows (60 min history, a prediction every 5 min) and tune one threshold per alarm on the last 20% of training. The ASPIRE-style Transformer is trained for 25 epochs on every training window with class-weighted loss. Means over seeds; std across seeds.

| asset | method | macro_f1 | macro_f1_std | macro_pr_auc | event_recall | median_lead_min | false_alerts_per_day |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_inverter | Raw XGB (ablation) | 0.588 | 0.059 | 0.672 | 0.806 | 27.167 | 2.042 |
| hybrid_inverter | ASPIRE-style Transformer (per-unit) | 0.425 | 0.021 | 0.550 | 0.970 | 27.500 | 5.350 |
| hybrid_inverter | ASPIRE-style Transformer (raw) | 0.385 | 0.021 | 0.442 | 0.883 | 28.000 | 4.917 |
| hybrid_inverter | Ours (per-unit XGB) | 0.645 | 0.059 | 0.712 | 0.869 | 27.667 | 2.708 |
| pump_drive | Raw XGB (ablation) | 0.500 | 0.024 | 0.617 | 0.744 | 18.250 | 1.083 |
| pump_drive | ASPIRE-style Transformer (per-unit) | 0.332 | 0.002 | 0.458 | 0.814 | 27.333 | 5.158 |
| pump_drive | ASPIRE-style Transformer (raw) | 0.215 | 0.010 | 0.305 | 0.856 | 27.250 | 8.967 |
| pump_drive | Ours (per-unit XGB) | 0.572 | 0.006 | 0.652 | 0.765 | 23.917 | 0.783 |

## E2 - Unseen power ratings (profile swap only, no retraining)

Macro-F1. Smaller = 3 kW / 3 HP, larger = 10 kW / 10 HP. Drop = trained minus unseen (lower is better).

| asset | method | trained rating | smaller rating | larger rating | drop_smaller | drop_larger |
| --- | --- | --- | --- | --- | --- | --- |
| hybrid_inverter | Raw XGB (ablation) | 0.588 | 0.432 | 0.568 | 0.156 | 0.020 |
| hybrid_inverter | ASPIRE-style Transformer (per-unit) | 0.425 | 0.413 | 0.428 | 0.011 | -0.003 |
| hybrid_inverter | ASPIRE-style Transformer (raw) | 0.385 | 0.021 | 0.286 | 0.364 | 0.099 |
| hybrid_inverter | Ours (per-unit XGB) | 0.645 | 0.641 | 0.643 | 0.004 | 0.002 |
| pump_drive | Raw XGB (ablation) | 0.500 | 0.396 | 0.359 | 0.104 | 0.140 |
| pump_drive | ASPIRE-style Transformer (per-unit) | 0.332 | 0.325 | 0.319 | 0.007 | 0.012 |
| pump_drive | ASPIRE-style Transformer (raw) | 0.215 | 0.167 | 0.183 | 0.048 | 0.032 |
| pump_drive | Ours (per-unit XGB) | 0.572 | 0.591 | 0.575 | -0.020 | -0.003 |

![cross rating](fig_cross_rating.png)

## Per-alarm results for our method (trained rating, mean over seeds)

| asset | alarm | precision | recall | f1 | pr_auc | events | event_recall | median_lead_min | false_alerts_per_day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_inverter | INV-01 | 0.687 | 0.944 | 0.794 | 0.918 | 6.000 | 1.000 | 26.833 | 0.292 |
| hybrid_inverter | INV-02 | 0.549 | 0.954 | 0.695 | 0.805 | 6.000 | 1.000 | 27.833 | 0.425 |
| hybrid_inverter | INV-03 | 0.548 | 0.704 | 0.607 | 0.657 | 6.000 | 0.778 | 27.000 | 0.150 |
| hybrid_inverter | INV-05 | 0.828 | 0.500 | 0.597 | 0.792 | 6.000 | 0.722 | 24.000 | 0.050 |
| hybrid_inverter | INV-06 | 0.748 | 0.914 | 0.820 | 0.881 | 6.000 | 1.000 | 115.833 | 0.325 |
| hybrid_inverter | INV-07 | 0.389 | 0.361 | 0.369 | 0.413 | 6.000 | 0.667 | 48.000 | 0.433 |
| hybrid_inverter | INV-08 | 0.698 | 0.944 | 0.798 | 0.817 | 6.000 | 1.000 | 57.000 | 0.333 |
| hybrid_inverter | INV-09 | 0.788 | 0.958 | 0.863 | 0.938 | 6.000 | 1.000 | 57.667 | 0.158 |
| hybrid_inverter | INV-10 | 0.607 | 0.287 | 0.388 | 0.536 | 6.000 | 0.444 | 14.833 | 0.058 |
| hybrid_inverter | INV-11 | 0.662 | 0.778 | 0.713 | 0.788 | 6.000 | 0.944 | 24.500 | 0.142 |
| hybrid_inverter | INV-12 | 0.321 | 0.764 | 0.452 | 0.292 | 4.667 | 1.000 | 24.667 | 0.342 |
| pump_drive | DRV-01 | 0.433 | 0.067 | 0.115 | 0.198 | 5.333 | 0.200 | 14.250 | 0.025 |
| pump_drive | DRV-02 | 0.537 | 0.603 | 0.568 | 0.622 | 6.333 | 0.738 | 27.500 | 0.150 |
| pump_drive | DRV-03 | 0.649 | 0.533 | 0.584 | 0.634 | 4.333 | 0.783 | 21.667 | 0.092 |
| pump_drive | DRV-04 | 0.740 | 0.954 | 0.828 | 0.910 | 6.000 | 1.000 | 27.667 | 0.100 |
| pump_drive | DRV-05 | 0.509 | 0.717 | 0.594 | 0.611 | 3.333 | 1.000 | 23.667 | 0.200 |
| pump_drive | DRV-06 | 0.906 | 0.675 | 0.772 | 0.817 | 6.333 | 0.944 | 20.167 | 0.033 |
| pump_drive | DRV-07 | 0.763 | 0.754 | 0.755 | 0.827 | 7.333 | 0.952 | 24.500 | 0.133 |
| pump_drive | DRV-08 | 0.690 | 0.259 | 0.360 | 0.595 | 6.000 | 0.500 | 15.167 | 0.050 |

![lead time](fig_lead_time.png)

![example](fig_example_INV-11.png)

## E3 - Forecast Agent (60-minute horizon)

Mean absolute error in per-unit:

| asset | signal | persistence | seasonal | trend |
| --- | --- | --- | --- | --- |
| hybrid_inverter | battery_soc | 0.036 | 0.012 | 0.015 |
| hybrid_inverter | heatsink_temp | 0.025 | 0.024 | 0.024 |
| hybrid_inverter | output_power | 0.049 | 0.052 | 0.059 |
| pump_drive | heatsink_temp | 0.028 | 0.028 | 0.045 |
| pump_drive | output_power | 0.068 | 0.061 | 0.146 |
| pump_drive | tank_level | 0.046 | 0.023 | 0.041 |

Time to trip, forecast made 30 minutes before each thermal alarm (caught = forecast crosses the limit within 60 min):

| asset | alarm | method | events | caught_rate | abs_err_min |
| --- | --- | --- | --- | --- | --- |
| hybrid_inverter | INV-11 | persistence | 6 | 0.000 |  |
| hybrid_inverter | INV-11 | seasonal | 6 | 0.000 |  |
| hybrid_inverter | INV-11 | trend | 6 | 0.833 | 4.600 |
| pump_drive | DRV-04 | persistence | 6 | 0.000 |  |
| pump_drive | DRV-04 | seasonal | 6 | 0.000 |  |
| pump_drive | DRV-04 | trend | 6 | 0.833 | 12.400 |

## E4 - Advisor Agent: correct manual entry per alarm event

| asset | ours_hit1 | ours_hit3 | readme_rag_hit1 | readme_rag_hit3 | events |
| --- | --- | --- | --- | --- | --- |
| hybrid_inverter | 0.988 | 1.000 | 0.092 | 0.257 | 584 |
| pump_drive | 0.910 | 0.998 | 0.114 | 0.385 | 413 |

Ours routes the Early-Warning agent's top alarm codes to their knowledge-base entries. README-style RAG queries the knowledge base with the raw sensor values, as the reference project does.

## Scope compared with the ASPIRE baseline

| Item | ASPIRE (as reproduced in Phase I) | This framework |
| --- | --- | --- |
| Assets | Metro air compressor (MetroPT-3) | Hybrid inverter, solar pump drive |
| Alarm types predicted | 1 failure category | 19 |
| Power ratings | 1 | 3 per asset type, by profile |
| Fix guidance | None | Cited checklist per alarm |

## Limitations

- Faults are injected into synthetic healthy telemetry; thresholds are placeholders until replaced with datasheet values.
- Per-unit features remove rating scale by design in this twin; real cross-rating data is needed to confirm the gain.
- Validate on real labelled faults (GPVS-Faults, inverter-driven PMSM dataset) with `scripts/validate_real.py` before claiming field performance.
- The knowledge base is authored from general practice; replace each source with the manual page of the reference device.
- ASPIRE-style baseline re-implements the Prediction Agent's Transformer only, not the full ASPIRE system.