"""
Decision-theoretic cost framework — paper Section III-F / IV-H, Eq. 23-28.
Adapts Hughes, Perinpanayagam & Ball, "Cost-efficiency and cost-effectiveness of
XAI in predictive maintenance," IEEE Access, vol. 13, pp. 151664-151670, 2025,
DOI: 10.1109/ACCESS.2025.3601385.
"""
import numpy as np


def fmcr(c_r: float, c_f: float) -> float:
    """Failure Mitigation Cost Ratio, Eq. 23."""
    return c_r / c_f


def is_cost_efficient(fmcr_value: float, precision: float) -> bool:
    """Eq. 24."""
    return fmcr_value < precision


def falsehood_ratio(fp: int, fn: int, tp: int) -> float:
    return fp / (fn + tp) if (fn + tp) else 0.0


def cer(recall: float, fr: float, fmcr_value: float) -> float:
    """Cost-Effectiveness Ratio relative to a perfect predictor, Eq. 25."""
    return recall - fr * (fmcr_value / (1 - fmcr_value))


def cer_with_expert_agent(recall: float, fr: float, fmcr_value: float, alpha: float) -> float:
    """Eq. 26 — alpha in [0,1]; alpha=1 reduces to Eq. 25 (no Expert Agent benefit)."""
    return (recall - alpha * fmcr_value * (recall + fr)) / (1 - fmcr_value)


def incremental_utility(delta_tp: int, delta_fp_avoided: int, c_f: float, c_r: float) -> float:
    """Eq. 27-28: delta_U = delta_tp*(C_F - C_R) + delta_fp_avoided*C_R."""
    return delta_tp * (c_f - c_r) + delta_fp_avoided * c_r


def sweep_fmcr(precision: float, recall: float, fp: int, fn: int, tp: int,
               fmcr_range=(0.05, 0.80), n_points: int = 20,
               alphas=(1.0, 0.90, 0.75, 0.60)) -> dict:
    fr = falsehood_ratio(fp, fn, tp)
    fmcr_values = np.linspace(*fmcr_range, n_points)
    results = {"fmcr": fmcr_values.tolist(), "cost_efficient": [], "cer": {a: [] for a in alphas}}
    for f in fmcr_values:
        results["cost_efficient"].append(bool(is_cost_efficient(f, precision)))
        for a in alphas:
            if a == 1.0:
                results["cer"][a].append(cer(recall, fr, f))
            else:
                results["cer"][a].append(cer_with_expert_agent(recall, fr, f, a))
    return results
