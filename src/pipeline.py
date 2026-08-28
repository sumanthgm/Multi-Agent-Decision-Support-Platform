"""
End-to-end ASPIRE reproduction pipeline. Run after `data_pipeline.py` has
produced data/processed/windows.npz.

    python src/pipeline.py --holdout_windows 2000

This trains all agents from scratch (no pretrained weights are published by the
authors) and reports metrics in the paper's exact tables' shape (Table 5, 6, 7).
Expect your absolute numbers to differ from the paper (see README §"Honesty note").
"""
import argparse
import yaml
import numpy as np

from sensor_agent import train_sensor_autoencoder, SensorAgent
from master_aggregator import MasterSensorAggregator
from adaptive_window_agent import build_gdrnet_mlp, build_next_step_predictor, AdaptiveWindowAgent
from prediction_agent import build_prediction_agent, train_prediction_agent, predict_probabilities
from decision_agent import DecisionAgent
import evaluate as ev


def load_config(path="configs/policy_params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def main(holdout_windows: int):
    cfg = load_config()
    data = np.load("data/processed/windows.npz", allow_pickle=True)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    X_holdout, y_holdout = data["X_holdout"][:holdout_windows], data["y_holdout"][:holdout_windows]
    feature_cols = list(data["feature_cols"])
    n_sensors = len(feature_cols)
    window_len = X_train.shape[1]

    print(f"train={X_train.shape} test={X_test.shape} holdout={X_holdout.shape}")

    # ---- 1. Prediction Agent (the frozen Transformer baseline) ----
    print("Training Prediction Agent (Transformer)...")
    pred_cfg = cfg["prediction_agent_transformer"]
    prediction_model = build_prediction_agent(
        window_len, n_sensors, pred_cfg["model_dim"], pred_cfg["attention_heads"],
        pred_cfg["feed_forward_dim"], pred_cfg["dropout"], pred_cfg["num_encoder_blocks"])
    train_prediction_agent(prediction_model, X_train, y_train, X_test, y_test)

    pred_probs_holdout = predict_probabilities(prediction_model, X_holdout)
    transformer_argmax = np.argmax(pred_probs_holdout, axis=1)

    # ---- 2. Sensor Agents (one LSTM-AE per sensor, trained on normal windows only) ----
    print("Training Sensor Agents (this trains N=%d autoencoders)..." % n_sensors)
    normal_mask = y_train == 0
    sensor_agents = []
    sa_cfg = cfg["sensor_agent"]
    for i, name in enumerate(feature_cols):
        x_1d = X_train[normal_mask, :, i][..., None]
        model = train_sensor_autoencoder(
            x_1d, window_len, sa_cfg["train"]["batch_size"],
            sa_cfg["train"]["max_epochs"], sa_cfg["train"]["early_stopping_patience"])
        agent = SensorAgent(model, sa_cfg, sensor_name=name)
        val_errors = np.mean((model.predict(x_1d[:200], verbose=0) - x_1d[:200]) ** 2, axis=(1, 2))
        agent.initialize_baseline(val_errors)
        sensor_agents.append(agent)
    aggregator = MasterSensorAggregator(cfg["master_sensor_aggregator"])

    # ---- 3. Adaptive Window Agent (GDRNet MLP + NSP LSTM) ----
    print("Training Adaptive Window Agent...")
    awa_cfg = cfg["adaptive_window_agent"]
    flat_dim = window_len * n_sensors
    gdrnet = build_gdrnet_mlp(flat_dim)
    # GDRNet target: paper omits exact ground-truth construction; a reasonable
    # proxy target used here is a rolling-variance-derived "informative window"
    # length — CALIBRATE against your own criterion / GDRNet paper (ref [12]).
    proxy_targets = np.clip(
        50 + 20 * np.tanh(np.std(X_train.reshape(len(X_train), -1), axis=1) - 0.5), 2, window_len)
    gdrnet.fit(X_train.reshape(len(X_train), -1), proxy_targets,
               batch_size=awa_cfg["gdrnet_mlp"]["train"]["batch_size"],
               epochs=awa_cfg["gdrnet_mlp"]["train"]["max_epochs"], verbose=2,
               validation_split=0.1,
               callbacks=[__import__("tensorflow").keras.callbacks.EarlyStopping(
                   patience=awa_cfg["gdrnet_mlp"]["train"]["early_stopping_patience"],
                   restore_best_weights=True)])

    nsp = build_next_step_predictor(n_sensors)
    n_sample = min(awa_cfg["next_step_predictor"]["train_windows_sampled"], len(X_train) - 1)
    idx = np.random.choice(len(X_train) - 1, n_sample, replace=False)
    nsp.fit(X_train[idx], X_train[idx + 1][:, -1, :],
            batch_size=awa_cfg["next_step_predictor"]["train"]["batch_size"],
            epochs=awa_cfg["next_step_predictor"]["train"]["epochs"], verbose=2)

    window_agent = AdaptiveWindowAgent(gdrnet, nsp, awa_cfg["runtime"])
    baseline_errors = np.mean((nsp.predict(X_train[:500], verbose=0) - X_train[:500, -1, :]) ** 2, axis=1)
    baseline_windows = gdrnet.predict(X_train[:500].reshape(500, -1), verbose=0).flatten()
    window_agent.fit_baseline(baseline_errors, baseline_windows)

    # ---- 4. Decision Agent — streaming arbitration over the holdout stream ----
    print("Running Decision Agent over holdout stream...")
    decision_agent = DecisionAgent(cfg=cfg)
    aspire_decisions = []
    for t in range(len(X_holdout)):
        sensor_outs = [agent.step(X_holdout[t, :, i], t) for i, agent in enumerate(sensor_agents)]
        sensor_agg = aggregator.aggregate(sensor_outs, cfg["sensor_aggregation"]["top_k"])
        w_out = window_agent.step(X_holdout[t].flatten(), X_holdout[t])
        out = decision_agent.decide(sensor_agg, w_out, pred_probs_holdout[t])
        cls = 2 if out["final_failure"] else (1 if out["final_warning"] else 0)
        aspire_decisions.append(cls)
    aspire_decisions = np.array(aspire_decisions)

    # ---- 5. Evaluation (Table 5, 6, 7 shape) ----
    warn_metrics = ev.evaluate_warning(y_holdout, transformer_argmax, aspire_decisions)
    ev.print_report(warn_metrics, "Early-warning prediction (Class 1 vs rest)")
    fault_metrics = ev.evaluate_fault_detection(y_holdout, transformer_argmax, aspire_decisions)
    ev.print_report(fault_metrics, "Fault detection (Class 2 vs rest)")
    buckets = ev.agreement_buckets(
        (transformer_argmax == 1).astype(int), (aspire_decisions == 1).astype(int), y_holdout)
    print("--- Agreement buckets (Table 7 shape) ---")
    for k, v in buckets.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout_windows", type=int, default=2000)
    args = ap.parse_args()
    main(args.holdout_windows)
