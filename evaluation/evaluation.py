"""
evaluation.py
-------------
All evaluation metrics used in the paper:
  - Expected Calibration Error (ECE)
  - Adaptive Calibration Error (ACE)
  - Brier Score
  - Coverage and set-size for conformal
  - Decision utility / net benefit
  - Statistical testing (Wilcoxon + Holm correction)
"""

import numpy as np
from scipy import stats as scipy_stats
from itertools import combinations


# ─────────────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────────────

def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error with equal-width bins.
    Lower is better.
    """
    probs  = np.asarray(probs,  dtype=float)
    labels = np.asarray(labels, dtype=int)
    bins   = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0
    n = len(probs)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        acc  = labels[mask].mean()
        conf = probs[mask].mean()
        ece_val += (mask.sum() / n) * abs(acc - conf)
    return float(ece_val)


def ace(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    Adaptive Calibration Error with equal-frequency bins.
    Lower is better.
    """
    probs  = np.asarray(probs,  dtype=float)
    labels = np.asarray(labels, dtype=int)
    n = len(probs)
    order  = np.argsort(probs)
    p_sort = probs[order]
    l_sort = labels[order]

    bin_size = n // n_bins
    ace_val  = 0.0
    for i in range(n_bins):
        start = i * bin_size
        end   = (i + 1) * bin_size if i < n_bins - 1 else n
        if end <= start:
            continue
        p_bin = p_sort[start:end]
        l_bin = l_sort[start:end]
        acc   = l_bin.mean()
        conf  = p_bin.mean()
        ace_val += (len(p_bin) / n) * abs(acc - conf)
    return float(ace_val)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean squared error between probabilities and binary labels."""
    probs  = np.asarray(probs,  dtype=float)
    labels = np.asarray(labels, dtype=float)
    return float(np.mean((probs - labels) ** 2))


def reliability_data(probs: np.ndarray, labels: np.ndarray,
                     n_bins: int = 15) -> dict:
    """
    Returns bin midpoints, mean confidence, accuracy, and count per bin
    for plotting reliability diagrams.
    """
    probs  = np.asarray(probs,  dtype=float)
    labels = np.asarray(labels, dtype=int)
    bins   = np.linspace(0.0, 1.0, n_bins + 1)
    mids, conf_vals, acc_vals, counts = [], [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        mids.append((lo + hi) / 2)
        conf_vals.append(probs[mask].mean())
        acc_vals.append(labels[mask].mean())
        counts.append(mask.sum())
    return {"midpoints": mids, "confidence": conf_vals,
            "accuracy": acc_vals, "counts": counts}


# ─────────────────────────────────────────────────────────────────────────────
# Decision Utility
# ─────────────────────────────────────────────────────────────────────────────

def net_benefit(probs: np.ndarray, labels: np.ndarray,
                threshold: float) -> float:
    """
    Net Benefit at a single threshold τ:
      NB(τ) = TPR * P/N − FPR * (1-P)/N * τ/(1-τ)
    where P = number of positives, N = total.
    """
    probs  = np.asarray(probs,  dtype=float)
    labels = np.asarray(labels, dtype=int)
    preds  = (probs >= threshold).astype(int)
    n      = len(labels)
    p      = labels.sum()

    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()

    # Avoid division by zero
    if threshold >= 1.0:
        return 0.0
    odds = threshold / (1.0 - threshold)
    return float(tp / n - fp / n * odds)


def decision_curve(probs: np.ndarray, labels: np.ndarray,
                   thresholds: np.ndarray | None = None) -> dict:
    """
    Full decision curve: net benefit across a range of thresholds.
    Also computes the 'treat all' and 'treat none' baselines.
    Returns dict with keys: thresholds, net_benefit, treat_all, treat_none.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)
    n = len(labels)
    p = labels.sum()
    prev = p / n

    nb_model   = [net_benefit(probs, labels, t) for t in thresholds]
    treat_all  = [prev - (1 - prev) * t / (1 - t) for t in thresholds]
    treat_none = [0.0] * len(thresholds)

    return {
        "thresholds":  thresholds,
        "net_benefit": np.array(nb_model),
        "treat_all":   np.array(treat_all),
        "treat_none":  np.array(treat_none),
    }


def threshold_metrics(probs: np.ndarray, labels: np.ndarray,
                      threshold: float = 0.5) -> dict:
    """TPR, FPR, precision, F1, and net benefit at a fixed threshold."""
    probs  = np.asarray(probs,  dtype=float)
    labels = np.asarray(labels, dtype=int)
    preds  = (probs >= threshold).astype(int)

    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()

    tpr = tp / (tp + fn + 1e-9)
    fpr = fp / (fp + tn + 1e-9)
    ppv = tp / (tp + fp + 1e-9)
    f1  = 2 * tpr * ppv / (tpr + ppv + 1e-9)

    return {"tpr": float(tpr), "fpr": float(fpr),
            "precision": float(ppv), "f1": float(f1),
            "net_benefit": net_benefit(probs, labels, threshold)}


# ─────────────────────────────────────────────────────────────────────────────
# Discriminative performance
# ─────────────────────────────────────────────────────────────────────────────

def auroc(probs: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, probs))


def auprc(probs: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(labels, probs))


# ─────────────────────────────────────────────────────────────────────────────
# Statistical testing
# ─────────────────────────────────────────────────────────────────────────────

def wilcoxon_pairwise(metric_dict: dict[str, list[float]],
                      alternative: str = "two-sided") -> dict:
    """
    Perform pairwise Wilcoxon signed-rank tests between all methods,
    then apply Holm-Bonferroni correction.

    Parameters
    ----------
    metric_dict : {method_name: [metric values across seeds/folds]}
    alternative : 'two-sided' | 'less' | 'greater'

    Returns
    -------
    dict with keys (method_a, method_b) → {statistic, p_raw, p_adj, significant}
    """
    methods = list(metric_dict.keys())
    pairs   = list(combinations(methods, 2))

    raw_p = {}
    stats_dict = {}
    for a, b in pairs:
        va = np.array(metric_dict[a])
        vb = np.array(metric_dict[b])
        stat, p = scipy_stats.wilcoxon(va, vb, alternative=alternative,
                                       zero_method="wilcox")
        raw_p[(a, b)]   = p
        stats_dict[(a, b)] = stat

    # Holm-Bonferroni correction
    pair_list = list(raw_p.keys())
    p_values  = np.array([raw_p[k] for k in pair_list])
    p_adj     = _holm_bonferroni(p_values)

    results = {}
    for k, p_a in zip(pair_list, p_adj):
        results[k] = {
            "statistic":   stats_dict[k],
            "p_raw":       raw_p[k],
            "p_adj":       p_a,
            "significant": p_a < 0.05,
        }
    return results


def _holm_bonferroni(p_values: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni correction. Returns adjusted p-values."""
    n = len(p_values)
    order  = np.argsort(p_values)
    p_adj  = np.zeros(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adjusted = p_values[idx] * (n - rank)
        running_max = max(running_max, adjusted)
        p_adj[idx] = min(running_max, 1.0)
    return p_adj


# ─────────────────────────────────────────────────────────────────────────────
# Comprehensive evaluation summary
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all(probs: np.ndarray, labels: np.ndarray,
                 conformal=None) -> dict:
    """
    Return a full evaluation dictionary for a single method + test set.

    Parameters
    ----------
    probs     : predicted probabilities (N,)
    labels    : binary ground truth (N,)
    conformal : optional fitted SplitConformalClassifier instance
    """
    out = {
        "auroc":  auroc(probs, labels),
        "auprc":  auprc(probs, labels),
        "ece":    ece(probs, labels),
        "ace":    ace(probs, labels),
        "brier":  brier_score(probs, labels),
    }
    if conformal is not None:
        out["coverage"]     = conformal.coverage(probs, labels)
        out["avg_set_size"] = conformal.avg_set_size(probs)
        out["cov_violation"]= conformal.coverage_violation(probs, labels)

    # Decision metrics at clinically common threshold
    for tau in [0.2, 0.3, 0.5]:
        m = threshold_metrics(probs, labels, tau)
        out[f"nb_{tau}"]  = m["net_benefit"]
        out[f"tpr_{tau}"] = m["tpr"]
        out[f"fpr_{tau}"] = m["fpr"]

    return out
