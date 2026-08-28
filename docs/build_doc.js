const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, AlignmentType, ImageRun, PageBreak,
  LevelFormat, convertInchesToTwip, PageOrientation
} = require("docx");

const ACCENT = "2C5F8A";
const LIGHT = "EAF2FB";
const WARN = "B8860B";

// ---------- helpers ----------
function h1(text) { return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 300, after: 150 } }); }
function h2(text) { return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 220, after: 100 } }); }
function h3(text) { return new Paragraph({ text, heading: HeadingLevel.HEADING_3, spacing: { before: 160, after: 80 } }); }
function p(text, opts = {}) {
  return new Paragraph({ spacing: { after: 120 }, children: [new TextRun({ text, ...opts })] });
}
function pr(runs, opts = {}) { return new Paragraph({ spacing: { after: 120 }, ...opts, children: runs }); }
function italic(text) { return new TextRun({ text, italics: true }); }
function bold(text) { return new TextRun({ text, bold: true }); }
function bullet(text, level = 0) {
  return new Paragraph({ text, bullet: { level }, spacing: { after: 60 } });
}
function numbered(text, ref = "num1") {
  return new Paragraph({ text, numbering: { reference: ref, level: 0 }, spacing: { after: 60 } });
}
function note(text) {
  return new Paragraph({
    spacing: { after: 140 },
    children: [new TextRun({ text, italics: true, color: "555555", size: 20 })],
  });
}
function tag(text, color) {
  return new TextRun({ text: " " + text + " ", bold: true, color: "FFFFFF", shading: { type: ShadingType.CLEAR, fill: color } });
}

function cell(text, opts = {}) {
  const { width = 2000, bold: b = false, shade = null, fontSize = 20 } = opts;
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: shade ? { type: ShadingType.CLEAR, fill: shade } : undefined,
    margins: { top: 60, bottom: 60, left: 80, right: 80 },
    children: [new Paragraph({ children: [new TextRun({ text: String(text), bold: b, size: fontSize })] })],
  });
}

function makeTable(headers, rows, widths) {
  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map((hd, i) => cell(hd, { width: widths[i], bold: true, shade: LIGHT })),
  });
  const bodyRows = rows.map(r => new TableRow({
    children: r.map((c, i) => cell(c, { width: widths[i] })),
  }));
  return new Table({
    width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    columnWidths: widths,
    rows: [headerRow, ...bodyRows],
  });
}

// ---------- content ----------

const titlePage = [
  new Paragraph({ spacing: { before: 1200, after: 200 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "ASPIRE Reproduction Project", bold: true, size: 56, color: ACCENT })] }),
  new Paragraph({ spacing: { after: 400 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "An Agentic Decision System for Early Equipment Failure\nPrediction and Intervention in Industrial IIoT", size: 30 })] }),
  new Paragraph({ spacing: { after: 100 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "End-to-End Rebuild Plan, Architecture Explainer, and Reference Implementation", italics: true, size: 24 })] }),
  new Paragraph({ spacing: { before: 600, after: 60 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Source paper:", bold: true, size: 20 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 20 },
    children: [new TextRun({ text: "S. Guha and A. Datta, \"ASPIRE: An Agentic Decision System for Early Equipment Failure Prediction and Intervention in Industrial IIoT,\"", size: 20 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 20 },
    children: [new TextRun({ text: "IEEE Access, vol. 14, pp. 105006\u2013105031, 2026.", size: 20 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 20 },
    children: [new TextRun({ text: "DOI: 10.1109/ACCESS.2026.3711010  \u00b7  Open Access, CC BY 4.0", size: 20 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 400 },
    children: [new TextRun({ text: "Received 3 Jun 2026 \u00b7 Accepted 3 Jul 2026 \u00b7 Published 7 Jul 2026", size: 18, color: "666666" })] }),
  new Paragraph({ children: [new PageBreak()] }),
];

const integrityNotice = [
  h1("0. Integrity & Citation Notice \u2014 Read First"),
  pr([bold("This document was produced by an AI assistant (Claude) using the actual IEEE Access PDF supplied by the user "),
      new TextRun("as its only source for ASPIRE-specific facts. Every architectural detail, formula, hyperparameter, and metric below that is attributed to the paper was transcribed from that PDF, not recalled from memory or invented.")]),
  p("Nonetheless, three categories of claim appear in this document and are colour/label-coded throughout so you can cite them correctly in your own submission:"),
  new Table({
    width: { size: 9350, type: WidthType.DXA },
    columnWidths: [2200, 7150],
    rows: [
      new TableRow({ children: [cell("PAPER-REPORTED", { width: 2200, bold: true, shade: "D9EAD3" }), cell("A number, formula, or design choice copied directly from the paper's text, tables, or figures. Safe to cite as \u201cGuha & Datta (2026) report...\u201d", { width: 7150 })] }),
      new TableRow({ children: [cell("REPRODUCTION (THIS RUN)", { width: 2200, bold: true, shade: "FFF2CC" }), cell("A number you will generate by actually running the code in this package. Will differ from the paper \u2014 report it as your own reproduction, not the paper's result.", { width: 7150 })] }),
      new TableRow({ children: [cell("ENGINEERING FILL-IN", { width: 2200, bold: true, shade: "F4CCCC" }), cell("A detail the paper does not fully specify (e.g., exact \u03b1/\u03b2 weight values beyond Table 3, GDRNet's exact training target). Filled in with a documented, defensible default \u2014 you should calibrate it yourself and say so.", { width: 7150 })] }),
    ],
  }),
  p(""),
  pr([bold("Do not present REPRODUCTION or ENGINEERING FILL-IN values as the paper's own results in your submission. "),
      new TextRun("Misattributing your own numbers (or an AI-invented number) to a published paper is a citation integrity issue, and this document is built specifically to prevent that.")]),
];

const execSummary = [
  h1("1. Executive Summary"),
  p("ASPIRE reframes early industrial fault prediction from a purely predictive modelling problem into a decision-making problem under uncertainty. Instead of directly equating a classifier's most probable class with an alert (the dominant \u201cmodel-centric\u201d approach), ASPIRE wraps a frozen Transformer classifier inside a deterministic, auditable Decision Agent that also consumes sensor-level anomaly/drift evidence and window-level temporal-context evidence before committing to a warning or failure alert."),
  p("On the MetroPT-3 real-world compressor dataset, this decision layer improved two-hour-ahead early-warning precision from 0.875 to 0.889, recall from 0.941 to 0.953, and F1 from 0.906 to 0.920 over the Transformer baseline, while leaving fault-detection performance completely unchanged \u2014 all without retraining the underlying model. The gains are shown (via ablation) to come from two independent, named mechanisms: margin-based gating (precision) and context-supported near-miss recovery (recall)."),
  p("This document gives you: (1) the full architecture explained agent-by-agent with its governing equations; (2) a phase-by-phase project plan from data download to final evaluation; (3) the exact policy parameters (Table 3) needed to configure the Decision Agent; (4) the paper's own reported results, clearly separated from what your own run will produce; (5) a working code skeleton (accompanying this document) implementing every agent; and (6) open-source references and integration guidance."),
];

const paperSummary = [
  h1("2. What the Paper Actually Claims"),
  h2("2.1 Problem framing"),
  p("Early fault prediction in IIoT predictive maintenance (PdM) is usually posed as an end-to-end learning problem: a model maps a fixed-length multivariate sensor window directly to a warning/failure label. The paper identifies four recurring weaknesses of this approach:"),
  bullet("High recall is often bought at the cost of excessive false alarms \u2192 operator fatigue, unsafe manual overrides."),
  bullet("Warning probabilities near the decision threshold are unstable under noise/drift, making a single fixed threshold brittle."),
  bullet("Post-hoc explainability highlights influential inputs but rarely gives actionable guidance (fault mechanism, recommended action)."),
  bullet("Prediction and decision-making are conflated: the arg-max class IS the alert, with no room to weigh contextual evidence (sensor anomalies, regime shift) or asymmetric costs."),
  p("ASPIRE's core move is to separate learning from authority: learning-based agents (autoencoders, an MLP, an LSTM forecaster, a Transformer) each produce evidence; only a deterministic, non-learned Decision Agent commits an alert, using explicit and auditable policy thresholds."),
  h2("2.2 The six agents"),
  bullet("Sensor Agents (\u00d7S) \u2014 local perception: one per sensor, flags point anomalies and distributional drift."),
  bullet("Master Sensor Aggregator Agent \u2014 turns per-sensor booleans into system-wide fractions/flags."),
  bullet("Adaptive Window Agent \u2014 global temporal-context agent; tracks whether the system's own temporal structure is stable."),
  bullet("Prediction Agent \u2014 the frozen Transformer 3-class classifier (Normal / Warning / Fault); this doubles as the paper's baseline."),
  bullet("Decision Agent \u2014 deterministic policy engine that arbitrates all of the above into a final alert (the paper's central contribution)."),
  bullet("Expert Agent \u2014 non-intervening LLM explainer, invoked only for LOW/MEDIUM warnings, never touches the automated decision."),
  h2("2.3 Contributions claimed by the authors (PAPER-REPORTED)"),
  numbered("A decision-centric heterogeneous multi-agent architecture with non-overlapping agent roles."),
  numbered("A deterministic Decision Agent with two independently-verifiable mechanisms: margin-based gating (precision 0.875\u21920.889) and context-supported near-miss recovery (recall 0.941\u21920.953)."),
  numbered("Decision-oriented drift modelling: WDI and FDI treated as first-class decision signals rather than only retraining triggers."),
  numbered("A non-intervening LLM-based Expert Agent producing structured, human-readable explanations and recommended actions."),
  numbered("A rigorous empirical evaluation on MetroPT-3 across holdout lengths of 2,000\u201330,000 windows, with agreement-bucket analysis, ablation, sensitivity analysis, temporal-stability analysis, and a cost-efficiency/cost-effectiveness analysis (adapting Hughes et al., 2025)."),
];

const architecture = [
  h1("3. System Architecture"),
  p("The figure below is a schematic reproduction of the paper's Figure 1, redrawn from the text description (perception \u2192 prediction \u2192 decision \u2192 explanation layers)."),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new ImageRun({ type: "png", data: fs.readFileSync(__dirname + "/architecture_diagram.png"), transformation: { width: 620, height: 350 } })],
  }),
  h2("3.1 Sensor Agent \u2014 local perception (Eq. 2\u20135)"),
  p("Each of the S sensors gets its own univariate LSTM autoencoder (encoder-decoder, latent dim 4, time-distributed linear output), trained ONLY on normal-labelled windows. At runtime:"),
  bullet("Reconstruction error e\u1d62_t = MSE(window, autoencoder(window))."),
  bullet("Anomaly flagged when e\u1d62_t > median(e) + k\u00b7MAD(e), with k=2.0 and a 5-step cooldown to prevent alert flooding (Eq. 3)."),
  bullet("Drift flagged via Jensen\u2013Shannon divergence between a baseline error histogram and a recent 100-step buffer (threshold 0.10), confirmed only after 3 consecutive votes, then a 100-step cooldown. KS-test is the documented fallback if histogram estimation is unreliable."),
  bullet("Confidence score C\u1d62_t = \u03c3(z\u1d62_t), a smooth sigmoid of distance beyond threshold (Eq. 4\u20135) \u2014 gives graded severity, not a binary flag."),
  bullet("Retraining is recommended only when \u22652 of {sustained anomaly rate >30% in last 50 steps, drift rate >10%, mean error >2\u00d7 baseline, stats stale >7 days} hold \u2014 conservative, multi-criteria."),
  note("PAPER FINDING: sensor-level drift in their MetroPT-3 run was confirmed AFTER fault onset (timestep ~922), not before \u2014 architecturally expected given the 3-vote + cooldown design. The paper is explicit that sensor agents are current-state corroborating evidence, NOT an early-warning source on their own (confirmed later by the ablation study, V2 = V1)."),
  h2("3.2 Master Sensor Aggregator Agent (Table 1, left column)"),
  p("Purely deterministic, no trainable parameters. Aggregates the boolean outputs of all Sensor Agents into: fraction of anomalous sensors, fraction drifting, fraction recommending retraining, plus three system-level boolean flags at fixed thresholds (0.30 anomaly, 0.20 drift, 0.15 retrain)."),
  h2("3.3 Adaptive Window Agent \u2014 global temporal context (Eq. 6\u201312)"),
  p("This is the most architecturally novel perception component. It runs two frozen sub-models:"),
  bullet("GDRNet \u2014 a pretrained MLP (widths 64/32/16/8, LeakyReLU 0.1, dropout 0.1 after layer 1) that predicts the optimal sliding-window length W\u0304_t for the current data. GDRNet itself is the authors' own prior IEEE Access 2025 paper (ref [12]) \u2014 this ASPIRE paper only reuses it and gives the wrapper's training recipe, not GDRNet's internal architecture."),
  bullet("Next-step predictor (M_NSP) \u2014 a 128-unit LSTM + 64-unit dense ReLU head forecasting the next multivariate timestep across all S sensors, trained on 50,000 randomly sampled training windows."),
  p("From these two models it derives four indices every step:"),
  bullet("FDS_t (Forecast Deviation Score) = (e_t \u2212 \u0113)/(MAD_e+\u03b5) \u2014 scale-invariant forecast-error deviation (Eq. 9)."),
  bullet("FDI_t (Forecast Drift Index) = JSD(baseline FDS distribution, recent FDS distribution) (Eq. 10)."),
  bullet("WDI_t (Window Drift Index) = JSD(baseline predicted-window-size distribution, recent) (Eq. 11) \u2014 treated as the DOMINANT drift signal."),
  bullet("WSS_t (Window Shift Score) = (W\u0304_t \u2212 \u03bc_W)/(\u03c3_W+\u03b5) \u2014 instantaneous deviation of predicted window size (Eq. 12)."),
  note("PAPER FINDING (ablation V3): removing this agent costs 0.004 F1 (0.920\u21920.916) with precision rising and recall falling \u2014 it is the necessary source of the paper's RECALL improvement (near-miss recovery cannot happen without its contextual drift/anomaly evidence, confirmed by V6 = V5 exactly)."),
  h2("3.4 Prediction Agent \u2014 the frozen Transformer (Eq. 13)"),
  p("2 Transformer encoder blocks (model dim 64, 4 attention heads, feed-forward dim 128, dropout 0.2), global average pooling, dense softmax \u2192 [P_normal, P_warn, P_fault]. Trained with sparse categorical cross-entropy, Adam(1e-3), early stopping, class weighting for label imbalance. Once trained it is FROZEN \u2014 ASPIRE never retrains or replaces it. This model's raw arg-max IS the paper's baseline (Table 5 \u201cTransformer\u201d row)."),
  h2("3.5 Decision Agent \u2014 the core contribution (Eq. 14\u201318, Algorithms 1 & 2, Table 3)"),
  p("Deterministic, policy-driven, fully described in Section 4 below."),
  h2("3.6 Expert Agent \u2014 non-intervening explanation (Eq. 19)"),
  p("Invoked ONLY when warning_level is LOW or MEDIUM (Algorithm 2). Packages the full decision context (final decisions, alert level, class probabilities, detection/drift risk, FDS/FDI/WSS/WDI, a short sensor snapshot, recent decision history from a SQLite EventStore) and sends it to gpt-4o-mini via the OpenAI Responses API with a strict JSON-only schema, 400 max output tokens, up to 2 retries, two-stage parse fallback. Its output \u2014 {summary, exp, likely_fault, recommended_action, severity} \u2014 is logged for human review and NEVER fed back into automated decisions."),
];

const decisionAgentDetail = [
  h1("4. Decision Agent \u2014 Algorithms 1 & 2 (full logic)"),
  p("At every timestep the Decision Agent receives four evidence streams (Table 1 of the paper): Sensor Agent outputs, Master Aggregator outputs, Adaptive Window Agent outputs, and Prediction Agent probabilities. It then runs the following, entirely deterministic, sequence."),
  h2("4.1 Pre-computations"),
  numbered("Sensor anomaly intensity: I_s^anom = \u03b1\u00b7anomaly_rate + (1\u2212\u03b1)\u00b7topK_strength (Eq. 14).", "num2"),
  numbered("Window anomaly intensity: I_w^anom = \u03b1_wss\u00b7\u03c3(|WSS|) + \u03b1_sev\u00b7(1\u2212e^\u2212sev) + \u03b1_fds\u00b7\u03c3(FDS), with \u03b1_wss \u226b \u03b1_sev > \u03b1_fds (Eq. 15\u201316).", "num2"),
  numbered("Extract probabilities and arg-max class c from the Prediction Agent.", "num2"),
  numbered("Detection risk R_det = clip[0,1](\u03b2_S\u00b7I_s^anom + \u03b2_W\u00b7I_w^anom) (Eq. 17).", "num2"),
  numbered("Drift risk R_drift = clip[0,1](max(WDI, I_s^drift)) \u2014 window drift dominates (Eq. 18).", "num2"),
  numbered("Secondary flags: final_anomaly \u2190 [R_det \u2265 \u03c4_det]; final_drift \u2190 [R_drift \u2265 \u03c4_drift].", "num2"),
  h2("4.2 Final Failure Logic (highest conservatism)"),
  p("A failure fires when EITHER the fault probability alone clears its threshold, OR the warning probability is high AND both detection and drift risk are simultaneously high \u2014 i.e. failure is never declared on learned prediction alone without corroborating present-tense evidence, except in the unambiguous high-probability case."),
  h2("4.3 Final Warning Logic (the asymmetric, novel part)"),
  bullet("Warning margin \u0394 = p_warn \u2212 p_normal."),
  bullet("Margin gate: \u0394 \u2265 \u0394_margin (calibrated to 0.0338 via PR-curve sweep on a pre-holdout split, maximizing F1 under a minimum-recall constraint \u2014 this suppresses boundary-adjacent, unstable warnings \u2192 precision gain)."),
  bullet("Transformer-warning: arg-max class is Warning AND margin gate passes."),
  bullet("Near-miss: arg-max class is Normal but the deficit (p_normal \u2212 p_warn) is small (< \u0394_deficit = 0.03)."),
  bullet("Warning evidence: a near-miss case additionally corroborated by final_drift OR final_anomaly \u2192 recall gain (this is where sensor/window evidence earns its keep)."),
  bullet("final_warning = (Transformer-warning OR warning_evidence) AND NOT final_failure."),
  h2("4.4 Warning Level \u2192 downstream routing (Algorithm 2)"),
  makeTable(
    ["Condition", "Alert level", "Action"],
    [
      ["final_failure", "CRITICAL", "SEND_ALERT(FIX)"],
      ["final_warning AND level=HIGH", "HIGH", "SEND_ALERT(PREDICTIVE_INTERVENTION)"],
      ["final_warning AND level\u2208{MEDIUM,LOW}", "MEDIUM", "INVOKE_EXPERT_AGENT()"],
      ["otherwise", "NONE", "monitor only"],
    ],
    [4300, 2000, 3050]
  ),
  h2("4.5 Table 3 \u2014 all published policy parameters"),
  makeTable(
    ["Category", "Parameter", "Value", "Meaning"],
    [
      ["Detection fusion", "w_sensor / w_window", "0.20 / 0.80", "Weight of sensor vs. window anomaly intensity"],
      ["Detection decision", "\u03c4_det", "0.50", "System-level anomaly threshold"],
      ["Drift fusion", "w_WDI / w_sensor-drift / w_FDI", "0.60 / 0.30 / 0.10", "Window drift dominates fusion"],
      ["Drift decision", "\u03c4_drift", "0.35", "System-level drift threshold"],
      ["Failure prediction", "\u03c4_fail / \u03c4_fail^crit", "0.50 / 0.80", "Failure / critical-failure thresholds"],
      ["Warning gating", "\u03c4_warn", "0.50", "Base warning probability threshold"],
      ["Warning gating", "\u0394_margin", "0.0338", "PR-swept warning margin gate"],
      ["Warning gating", "\u0394_deficit", "0.03", "Near-miss tolerance"],
      ["Cooldowns", "N_anom / N_drift", "5 / 100 steps", "Prevents repeated triggering"],
      ["Alert mapping", "\u03c4_low/med/high", "0.35 / 0.55 / 0.75", "Operational heuristics"],
      ["Sensor aggregation", "K (top-K)", "5", "Limits noisy-sensor dominance"],
    ],
    [2000, 2600, 2200, 2550]
  ),
  note("Not published beyond ordering constraints: exact \u03b1_wss/\u03b1_sev/\u03b1_fds values (only \u03b1_wss \u226b \u03b1_sev > \u03b1_fds is stated) and the exact \u0394_margin_high/\u0394_margin_low cut points used for the HIGH/MEDIUM/LOW warning-level split in Algorithm 1. These are ENGINEERING FILL-IN in the accompanying code (configs/policy_params.yaml) \u2014 calibrate them on your own pre-holdout split before trusting absolute numbers."),
];

const dataSection = [
  h1("5. Dataset \u2014 MetroPT-3"),
  p("A publicly available, real-world IIoT dataset (train compressor air-production unit), Davari et al. 2021, UCI ML Repository DOI 10.24432/C5VW3R. 1Hz logging by an onboard embedded device, 10-second sampling, 1,516,948 observations, Feb\u2013Aug 2020, 15 raw sensor channels."),
  h2("5.1 Sensor selection"),
  p("Granger-causality correlation analysis removes 3 channels: Pressure_switch, Caudal_impulses, Oil_level. The remaining 12 are used: TP2, TP3, H1, DV pressure, Reservoirs, Oil Temperature, Motor Current, COMP, DV electric, TOWERS, MPG, LPS."),
  h2("5.2 Cleaning"),
  bullet("Data before 1 April 2020 excluded (constant/unreliable values)."),
  bullet("\u201c?\u201d values \u2192 NaN \u2192 imputed via one-day seasonal lag."),
  bullet("Per-feature Min-Max scaling to [\u22121, 1], parameters stored per feature."),
  h2("5.3 Windowing & labelling"),
  bullet("Sliding windows W=100 (\u2248 1000s / 16.7 min), stride=1, window aligned to its final timestamp."),
  bullet("Point labels: fault-in-progress (inside a failure interval) and early-warning (within H=2h before a failure)."),
  bullet("Window label via any-point rule over {Normal(0), Warning(1), Fault(2)}; Fault overrides Warning."),
  h2("5.4 Failure events (Table 2, PAPER-REPORTED)"),
  makeTable(
    ["#", "Start", "End", "Type", "Severity"],
    [
      ["1", "2020-04-18 00:00", "2020-04-18 23:59", "Air leak", "High stress"],
      ["2", "2020-05-29 23:30", "2020-05-30 06:00", "Air leak", "High stress"],
      ["3", "2020-06-05 10:00", "2020-06-07 14:30", "Air leak", "High stress"],
      ["4", "2020-07-15 14:30", "2020-07-15 19:00", "Air leak", "High stress"],
    ],
    [900, 2500, 2500, 1900, 1900]
  ),
  h2("5.5 Chronology-aware, fault-aware split"),
  bullet("Train: everything before Failure 3's early-warning start."),
  bullet("Test: the entire Failure-3 interval (used for calibration only, never for final reporting)."),
  bullet("Holdout: everything after Failure 3, which contains all of Failure 4 \u2014 reserved exclusively for final evaluation, never touched during threshold calibration."),
  note("Exact window counts per split are not published beyond the aggregate 1,516,948-observation total and the 4 failure windows \u2014 only the split LOGIC is specified precisely enough to reproduce; your own counts will differ slightly depending on exact row alignment."),
];

const projectPlan = [
  h1("6. Project Plan \u2014 Phased Rebuild"),
  p("This mirrors exactly the pipeline order in the accompanying code package (src/*.py)."),
  makeTable(
    ["Phase", "Steps", "Output"],
    [
      ["0. Environment", "Install Python 3.10+, TensorFlow/Keras, NumPy/SciPy/scikit-learn, PyYAML, openai SDK (requirements.txt).", "Working env"],
      ["1. Data acquisition", "Download MetroPT-3 from UCI (id 791, DOI 10.24432/C5VW3R) or Kaggle mirror. Place as data/metropt3_raw.csv.", "Raw CSV"],
      ["2. Preprocessing", "Run src/data_pipeline.py: sensor selection, date filter, imputation, min-max scaling, windowing, labelling, chronology split.", "data/processed/windows.npz"],
      ["3. Sensor Agents", "Train one LSTM-AE per sensor on normal-only windows (src/sensor_agent.py). Initialize median/MAD baselines from validation errors.", "12 trained autoencoders"],
      ["4. Master Aggregator", "Deterministic \u2014 no training. Wraps the 12 Sensor Agents (src/master_aggregator.py).", "Aggregation function"],
      ["5. Adaptive Window Agent", "Train GDRNet-style MLP (window-size regressor) + LSTM next-step predictor (src/adaptive_window_agent.py). Fit baseline FDS/window distributions.", "Trained AWA + baseline stats"],
      ["6. Prediction Agent", "Train the 2-block Transformer classifier on Train, validate on Test, with class weighting (src/prediction_agent.py). THIS is your baseline.", "Frozen Transformer + Table 5 \u2018Transformer\u2019 row"],
      ["7. Decision Agent calibration", "PR-sweep \u0394_margin on the Test split (Fig. 4 equivalent); fix all Table 3 values; freeze before touching Holdout.", "configs/policy_params.yaml finalised"],
      ["8. Streaming evaluation", "Run the full pipeline over the Holdout stream (2k/10k/30k windows) producing Table 5/6/7/8 equivalents (src/evaluate.py).", "Your reproduction metrics"],
      ["9. Ablation", "Run V1\u2013V6 variants over cached perception outputs, no retraining (src/ablation.py) \u2192 Table 11 equivalent.", "Mechanism attribution"],
      ["10. Cost framework", "Apply Eq. 23\u201328 across an FMCR sweep (src/cost_framework.py) \u2192 Fig. 9 equivalent.", "Cost-efficiency/-effectiveness curves"],
      ["11. Expert Agent", "Wire up OpenAI gpt-4o-mini via src/expert_agent.py; invoke only on LOW/MEDIUM warnings; log to SQLite EventStore.", "Human-readable case studies (Table 9/10 equivalents)"],
      ["12. Write-up", "Compile your own Table 5\u201311 equivalents, clearly labelled REPRODUCTION (THIS RUN), and compare qualitatively (not as a replication claim) against the PAPER-REPORTED numbers in Section 7.", "Final report"],
    ],
    [1600, 5200, 2550]
  ),
];

const evaluationProtocol = [
  h1("7. Evaluation Protocol & Metrics"),
  p("Two independent one-vs-rest binary tasks (Eq. 20\u201322):"),
  bullet("Early-warning prediction: Class 1 (Warning) = positive, Classes {0,2} = negative."),
  bullet("Fault detection: Class 2 (Fault) = positive, Classes {0,1} = negative."),
  pr([bold("Precision = TP/(TP+FP)"), new TextRun("   \u00b7   "), bold("Recall = TP/(TP+FN)"), new TextRun("   \u00b7   "), bold("F1 = 2TP/(2TP+FP+FN)")]),
  h2("7.1 Agreement buckets (Table 7 logic)"),
  p("Warning outcomes are split by whether the Transformer (T) and Decision Agent (D) agree, isolating the decision layer's own contribution:"),
  bullet("A(Tw,Dw): both warn \u2014 consensus, should be reliable."),
  bullet("B(Tw,Dn): Transformer-only \u2014 suppressed by margin gating (should skew toward fault-in-progress/normal, i.e. correctly suppressed noise)."),
  bullet("C(Tn,Dw): Decision-Agent-only \u2014 near-miss recovery (should contain the recall gain, constrained to LOW/MEDIUM alert levels only)."),
  bullet("D(Tn,Dn): both silent \u2014 dominant bucket, confirms no unnecessary alerting."),
  h2("7.2 Statistical caveats the paper itself raises"),
  bullet("Standard bootstrap/permutation CIs are NOT reported \u2014 windows overlap 99/100 steps (stride=1) breaking i.i.d. assumptions, and warning/fault labels concentrate in one contiguous block per failure, so resampling would produce false precision."),
  bullet("Instead, the paper uses temporal-stability analysis across increasing holdout lengths (2k\u219210k\u219230k windows) as its reliability evidence \u2014 reproduce this the same way rather than trying to bootstrap your own CIs."),
];

const paperResults = [
  h1("8. Paper-Reported Results (for reference only \u2014 not to be reproduced verbatim)"),
  h2("8.1 Table 5 \u2014 core metrics, 2,000-window holdout (PAPER-REPORTED)"),
  makeTable(
    ["Task", "Model", "Precision", "Recall", "F1"],
    [
      ["Warning", "Transformer", "0.875", "0.941", "0.906"],
      ["Warning", "ASPIRE", "0.889", "0.953", "0.920"],
      ["Failure", "Transformer", "1.000", "0.940", "0.969"],
      ["Failure", "ASPIRE", "1.000", "0.940", "0.969"],
    ],
    [1600, 2400, 1783, 1783, 1784]
  ),
  h2("8.2 Table 6 \u2014 warning-class confusion counts, 2,000-window holdout (PAPER-REPORTED)"),
  makeTable(
    ["Method", "TP", "FP", "TN", "FN"],
    [["Transformer", "683", "98", "1176", "43"], ["ASPIRE", "691", "86", "1188", "35"]],
    [2400, 1737, 1737, 1737, 1739]
  ),
  h2("8.3 Table 8 \u2014 robustness across holdout lengths (PAPER-REPORTED)"),
  makeTable(
    ["Holdout size", "Method", "Precision", "Recall", "F1"],
    [
      ["2,000", "Transformer", "0.875", "0.941", "0.906"],
      ["2,000", "ASPIRE", "0.889", "0.953", "0.920"],
      ["10,000", "Transformer", "0.830", "0.941", "0.882"],
      ["10,000", "ASPIRE", "0.847", "0.953", "0.896"],
      ["30,000", "Transformer", "0.688", "0.941", "0.795"],
      ["30,000", "ASPIRE", "0.704", "0.953", "0.809"],
    ],
    [1900, 2400, 1683, 1683, 1684]
  ),
  h2("8.4 Table 11 \u2014 ablation, 2,000-window holdout (PAPER-REPORTED)"),
  makeTable(
    ["Variant", "Precision", "Recall", "F1"],
    [
      ["V0: Transformer baseline (argmax only)", "0.875", "0.941", "0.906"],
      ["V1: Full ASPIRE (reported)", "0.889", "0.953", "0.920"],
      ["V2: No sensor agents", "0.889", "0.953", "0.920"],
      ["V3: No window agent", "0.913", "0.920", "0.916"],
      ["V4: No margin gate", "0.875", "0.941", "0.906"],
      ["V5: No near-miss recovery", "0.914", "0.917", "0.915"],
      ["V6: Post-processing baseline", "0.914", "0.917", "0.915"],
    ],
    [4350, 1667, 1667, 1666]
  ),
  h2("8.5 Cost-efficiency findings (Section IV-H, PAPER-REPORTED)"),
  bullet("ASPIRE is cost-efficient (FMCR < Precision) for all FMCR < 0.889, vs. FMCR < 0.875 for the Transformer."),
  bullet("At FMCR=0.20, CER gain = 0.015 (ASPIRE 0.922 vs Transformer 0.907); at FMCR=0.50, gain widens to 0.028 (0.834 vs 0.806)."),
  bullet("Incremental utility \u0394U = 8\u00b7C_F + 4\u00b7C_R, positive for any C_F, C_R > 0 (8 missed warnings converted to true warnings, 12 false alarms avoided per 2,000 windows)."),
  note("These are the authors' own numbers, on their own train/test/holdout split, their own random seeds, and (for several parameters) calibration values not fully published. Reproducing the exact figures bit-for-bit is not realistically achievable from the paper text alone \u2014 treat Section 9 below as the honest expectation-setting for your own run."),
];

const reproExpectations = [
  h1("9. What to Realistically Expect From Your Own Reproduction"),
  p("Your own run (via the accompanying code) will almost certainly NOT match Table 5\u2013Table 11 numbers exactly. This is expected and should be reported honestly, not hidden. Reasons, in order of impact:"),
  numbered("Unpublished exact split boundaries \u2014 you will derive your own train/test/holdout row indices from the split LOGIC (Section 5.5); the paper's own row counts are not given.", "num3"),
  numbered("Unpublished \u03b1/\u03b2 weight values (Eq. 14\u201316) beyond an ordering constraint \u2014 you must calibrate these on your own pre-holdout split, and different calibration will shift the operating point.", "num3"),
  numbered("GDRNet's exact training target is defined in a separate prior paper (ref [12], IEEE Access 2025, DOI 10.1109/ACCESS.2025.11259050) not reproduced here \u2014 the accompanying code uses a documented proxy target you should replace once you've read that paper.", "num3"),
  numbered("Random seeds, exact Transformer weight initialization, and training-run stochasticity (the paper fixes seeds, but doesn't publish their values) \u2014 expect run-to-run variance even with identical code.", "num3"),
  numbered("The Expert Agent's LLM outputs (gpt-4o-mini) are non-deterministic across API calls/versions, and Table 9\u201310's case studies are illustrative, not target metrics to hit.", "num3"),
  p("A credible, honest reproduction target is: (a) confirm your Transformer baseline is \u201cstrong\u201d in the same qualitative sense (comparable precision/recall in the same ballpark for this task); (b) confirm the DIRECTION and RELATIVE SIZE of ASPIRE's gain over your own baseline (precision up, recall up, F1 up by roughly a similar order of magnitude \u2014 low single-digit percentage points); (c) confirm the ablation STORY replicates qualitatively \u2014 margin gate drives precision, window-agent context drives recall, sensor agents matter far less for early-warning than for fault detection."),
];

const toolsStack = [
  h1("10. Tools & Technology Stack"),
  makeTable(
    ["Component", "Tool / Library", "Role"],
    [
      ["Neural models", "TensorFlow / Keras", "LSTM autoencoders, MLP (GDRNet), LSTM next-step predictor, Transformer classifier"],
      ["Numerical / stats", "NumPy, SciPy, scikit-learn", "JS-divergence, KS-test fallback, class weighting, general array ops"],
      ["Data handling", "pandas", "CSV loading, resampling, time-based indexing"],
      ["Config", "PyYAML", "Table 3 policy parameters as a single source of truth"],
      ["Event logging", "sqlite3 (built-in)", "Append-only EventStore for decisions, expert outputs (auditability)"],
      ["Expert Agent LLM", "OpenAI Python SDK, gpt-4o-mini, Responses API", "Structured JSON explanations, non-intervening"],
      ["Training compute", "Google Colab, NVIDIA L4 GPU (paper); any CUDA-capable GPU works", "GPU-accelerated training/inference"],
      ["Visualization", "matplotlib", "PR-sweep curves, ablation bar charts, cost-effectiveness plots"],
    ],
    [2000, 3400, 3950]
  ),
  note("The paper implements its six agents as plain Python classes coordinated by direct function calls \u2014 it does NOT use an agent-orchestration framework such as LangGraph or CrewAI. If you want an orchestration layer for engineering convenience (state machine visualization, retries, tracing) you can wrap the same logic in one, but that would be an extension beyond what's described, not a reproduction of it."),
];

const references = [
  h1("11. Open-Source & Reference Integration Points"),
  p("No official ASPIRE code repository was published alongside the paper (checked as of this writing). The following are general-purpose open-source references for the individual techniques the paper describes \u2014 use them as implementation aids, not as \u201cthe\u201d ASPIRE repo:"),
  bullet("MetroPT-3 dataset (UCI): https://archive.ics.uci.edu/dataset/791/metropt+3+dataset \u2014 DOI 10.24432/C5VW3R"),
  bullet("GDRNet (the authors' own prior work, reused inside the Adaptive Window Agent): S. Guha and A. Datta, IEEE Access, vol. 13, pp. 200382\u2013200393, 2025, https://ieeexplore.ieee.org/document/11259050"),
  bullet("Hughes, Perinpanayagam & Ball \u2014 cost-efficiency/cost-effectiveness framework this paper adapts: IEEE Access, vol. 13, pp. 151664\u2013151670, 2025, DOI 10.1109/ACCESS.2025.3601385"),
  bullet("Keras LSTM-autoencoder anomaly-detection example (pattern for Sensor Agents): https://keras.io/examples/timeseries/timeseries_anomaly_detection/"),
  bullet("Keras Transformer time-series classification example (pattern for Prediction Agent): https://keras.io/examples/timeseries/timeseries_transformer_classification/"),
  bullet("scipy.spatial.distance.jensenshannon \u2014 used for FDI/WDI and sensor-drift JSD: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html"),
  bullet("scipy.stats.ks_2samp \u2014 KS-test drift-detection fallback"),
  bullet("OpenAI Python SDK (Expert Agent): https://github.com/openai/openai-python"),
  bullet("Prior MetroPT studies cited for cross-checking preprocessing (methodologically distinct tasks, not directly comparable per the paper itself): Barros et al. 2020 (rule-based APU monitoring); Davari et al. 2021 DSAA (autoencoder anomaly detection); Najjar et al. 2023 (Random Forest classification, 67:33 split)."),
  bullet("Optional agent-orchestration frameworks if you extend beyond the paper's plain-Python design: LangGraph (https://github.com/langchain-ai/langgraph), CrewAI (https://github.com/crewAIInc/crewAI)."),
  p(""),
  pr([italic("Citation caution: ")," ", italic("this document and I (the AI assistant) do not have live web/database access in this conversation. GitHub URLs above were not freshness-checked against the live web at generation time \u2014 verify each link resolves before citing it, and prefer the DOIs (which are stable identifiers from the paper's own reference list) over any secondary GitHub mirror.")]),
];

const limitations = [
  h1("12. Limitations (Paper's Own, Section VI)"),
  bullet("Single dataset (MetroPT-3), single fault type (air leak), across only 4 failure events \u2014 a dataset constraint, not a framework limitation, per the authors."),
  bullet("No evaluation under simulated sensor noise or missingness (dropouts, calibration drift)."),
  bullet("No full latency/scalability characterisation across varying sensor counts, hardware, or sampling rates."),
  bullet("No comparison against other decision-aware baselines (cost-sensitive classifiers, Bayesian risk-minimisation)."),
  bullet("Future work flagged by the authors: CMAPSS (turbofan) and PRONOSTIA (bearing) datasets, human-in-the-loop policy tuning, multi-horizon decision strategies."),
];

// numbering config for numbered() paragraphs
const numLevels = [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.START,
  style: { paragraph: { indent: { left: convertInchesToTwip(0.5), hanging: convertInchesToTwip(0.25) } } } }];
const numbering = {
  config: [
    { reference: "num1", levels: numLevels },
    { reference: "num2", levels: numLevels },
    { reference: "num3", levels: numLevels },
  ],
};

const doc = new Document({
  numbering,
  styles: {
    default: { document: { run: { font: "Calibri", size: 22 } } },
    heading1: { run: { color: ACCENT, size: 30, bold: true } },
    heading2: { run: { color: ACCENT, size: 26, bold: true } },
    heading3: { run: { color: "444444", size: 23, bold: true } },
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 } } }, // US Letter
    children: [
      ...titlePage, ...integrityNotice, ...execSummary, ...paperSummary,
      ...architecture, ...decisionAgentDetail, ...dataSection, ...projectPlan,
      ...evaluationProtocol, ...paperResults, ...reproExpectations, ...toolsStack,
      ...references, ...limitations,
    ],
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(__dirname + "/ASPIRE_Reproduction_Plan.docx", buf);
  console.log("Wrote ASPIRE_Reproduction_Plan.docx");
});
