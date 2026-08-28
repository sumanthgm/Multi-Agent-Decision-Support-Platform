"""
Evaluation protocol — paper Section III-E, IV-A, IV-B (Table 5, 6, 7, 8).

Point-based, one-vs-rest binary framing per Eq. 20-22:
  - Early-warning prediction: Class 1 (warning) positive vs {0,2} negative.
  - Fault detection: Class 2 (fault) positive vs {0,1} negative.

Also computes the four agreement buckets (Table 7) between the Transformer
baseline's raw argmax decision and ASPIRE's Decision Agent final decision.
"""
import numpy as np


def prf1(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> dict:
    tp = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
    fp = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
    fn = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))
    tn = int(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "precision": precision, "recall": recall, "f1": f1}


def evaluate_warning(y_true: np.ndarray, transformer_argmax: np.ndarray,
                      aspire_decisions: np.ndarray) -> dict:
    """y_true, transformer_argmax, aspire_decisions all in {0,1,2} (Normal/Warning/Fault)."""
    yt_bin = (y_true == 1).astype(int)
    t_bin = (transformer_argmax == 1).astype(int)
    a_bin = (aspire_decisions == 1).astype(int)
    return {
        "transformer": prf1(yt_bin, t_bin),
        "aspire": prf1(yt_bin, a_bin),
    }


def evaluate_fault_detection(y_true: np.ndarray, transformer_argmax: np.ndarray,
                              aspire_decisions: np.ndarray) -> dict:
    yt_bin = (y_true == 2).astype(int)
    t_bin = (transformer_argmax == 2).astype(int)
    a_bin = (aspire_decisions == 2).astype(int)
    return {
        "transformer": prf1(yt_bin, t_bin),
        "aspire": prf1(yt_bin, a_bin),
    }


def agreement_buckets(transformer_warns: np.ndarray, aspire_warns: np.ndarray,
                       y_true: np.ndarray) -> dict:
    """Table 7: A(Tw,Dw), B(Tw,Dn), C(Tn,Dw), D(Tn,Dn) with ground-truth composition."""
    buckets = {}
    masks = {
        "A_Tw_Dw": (transformer_warns == 1) & (aspire_warns == 1),
        "B_Tw_Dn": (transformer_warns == 1) & (aspire_warns == 0),
        "C_Tn_Dw": (transformer_warns == 0) & (aspire_warns == 1),
        "D_Tn_Dn": (transformer_warns == 0) & (aspire_warns == 0),
    }
    for name, mask in masks.items():
        n = int(mask.sum())
        if n == 0:
            buckets[name] = {"count": 0, "normal_pct": 0.0, "warning_pct": 0.0, "fault_pct": 0.0}
            continue
        yt = y_true[mask]
        buckets[name] = {
            "count": n,
            "normal_pct": float(np.mean(yt == 0) * 100),
            "warning_pct": float(np.mean(yt == 1) * 100),
            "fault_pct": float(np.mean(yt == 2) * 100),
        }
    return buckets


def print_report(metrics: dict, label: str):
    print(f"--- {label} ---")
    for model_name, m in metrics.items():
        print(f"  {model_name:12s} P={m['precision']:.3f} R={m['recall']:.3f} "
              f"F1={m['f1']:.3f}  (TP={m['TP']} FP={m['FP']} TN={m['TN']} FN={m['FN']})")
