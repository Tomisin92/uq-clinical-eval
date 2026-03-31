"""
run_experiments.py
------------------
Full experiment pipeline. Run this after preprocessing MIMIC-III data.

Usage:
    python run_experiments.py

Outputs (written to results/):
    results_raw.json    -- all metrics per seed per task per method
    results_summary.json -- mean ± std tables ready for LaTeX
    stats_tests.json    -- pairwise Wilcoxon + Holm results
"""

import os, sys, json, time
import numpy as np
import pandas as pd
import torch

# ── local imports ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from data.preprocess   import make_splits
from models.models     import BaseNet, BayesNet, mc_dropout_predict
from models.train      import train_basenet, train_bnn
from models.conformal  import SplitConformalClassifier
from evaluation.evaluation import evaluate_all, wilcoxon_pairwise

# ── config ───────────────────────────────────────────────────────────────────
PROC_DIR     = os.path.join(os.path.dirname(__file__), "data", "processed")
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

TASKS        = ["mortality", "readmission_30d"]
SEEDS        = [42, 43, 44, 45, 46]
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
HIDDEN       = (128, 64, 32)
DROPOUT      = 0.3
PRIOR_STD    = 1.0
LR           = 1e-3
EPOCHS       = 100
PATIENCE     = 10
ALPHA        = 0.10    # conformal miscoverage

print(f"Device: {DEVICE}")


def run_one_seed(task: str, seed: int) -> dict:
    """Run all three methods for one task/seed. Returns metrics dict."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # ── load data ────────────────────────────────────────────────────────────
    feat = pd.read_csv(os.path.join(PROC_DIR, f"{task}_features.csv"), low_memory=False)
    splits = make_splits(feat, label_col=task if task == "mortality" else "readmission_30d",
                         train_years=(2110, 2190), shift_years=(2191, 2214),
                         seed=seed)
    X_tr, y_tr = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]
    X_cal, y_cal = splits["X_cal"], splits["y_cal"]
    X_sh,  y_sh  = splits["X_shift"], splits["y_shift"]
    # Use validation set as IID test (different patients, same time period)
    X_iid, y_iid = X_val, y_val

    input_dim = X_tr.shape[1]

    def to_tensor(arr):
        return torch.tensor(arr, dtype=torch.float32).to(DEVICE)

    results = {}

    # ════════════════════════════════════════════════════════════════════════
    # MC Dropout
    # ════════════════════════════════════════════════════════════════════════
    print(f"    MC Dropout ...", end="", flush=True)
    mc_model = BaseNet(input_dim, HIDDEN, DROPOUT)
    train_basenet(mc_model, X_tr, y_tr, X_val, y_val,
                  EPOCHS, LR, patience=PATIENCE, device=DEVICE, verbose=False)
    mc_model.eval()

    def mc_probs(X):
        mean, _ = mc_dropout_predict(mc_model, to_tensor(X), T=50)
        return mean.cpu().numpy()

    iid_p = mc_probs(X_iid)
    sh_p  = mc_probs(X_sh)

    cp_mc = SplitConformalClassifier(alpha=ALPHA)
    cp_mc.calibrate(mc_probs(X_cal), y_cal)

    results["mc_dropout"] = {
        "iid":   evaluate_all(iid_p, y_iid, cp_mc),
        "shift": evaluate_all(sh_p,  y_sh,  cp_mc),
    }
    print(" done")

    # ════════════════════════════════════════════════════════════════════════
    # BNN
    # ════════════════════════════════════════════════════════════════════════
    print(f"    BNN ...", end="", flush=True)
    bnn_model = BayesNet(input_dim, HIDDEN, PRIOR_STD)
    train_bnn(bnn_model, X_tr, y_tr, X_val, y_val,
              EPOCHS, LR, patience=PATIENCE, device=DEVICE, verbose=False)

    def bnn_probs(X):
        mean, _ = bnn_model.predict(to_tensor(X), S=30)
        return mean.cpu().numpy()

    iid_p = bnn_probs(X_iid)
    sh_p  = bnn_probs(X_sh)

    cp_bnn = SplitConformalClassifier(alpha=ALPHA)
    cp_bnn.calibrate(bnn_probs(X_cal), y_cal)

    results["bnn"] = {
        "iid":   evaluate_all(iid_p, y_iid, cp_bnn),
        "shift": evaluate_all(sh_p,  y_sh,  cp_bnn),
    }
    print(" done")

    # ════════════════════════════════════════════════════════════════════════
    # Conformal Prediction (base model = deterministic BaseNet)
    # ════════════════════════════════════════════════════════════════════════
    print(f"    Conformal ...", end="", flush=True)
    cp_base = BaseNet(input_dim, HIDDEN, dropout=0.0)   # no dropout for base
    train_basenet(cp_base, X_tr, y_tr, X_val, y_val,
                  EPOCHS, LR, patience=PATIENCE, device=DEVICE, verbose=False)
    cp_base.eval()

    def det_probs(X):
        with torch.no_grad():
            return cp_base(to_tensor(X)).cpu().numpy()

    cp = SplitConformalClassifier(alpha=ALPHA)
    cp.calibrate(det_probs(X_cal), y_cal)

    iid_p = det_probs(X_iid)
    sh_p  = det_probs(X_sh)

    results["conformal"] = {
        "iid":   evaluate_all(iid_p, y_iid, cp),
        "shift": evaluate_all(sh_p,  y_sh,  cp),
    }
    print(" done")

    return results


def aggregate_results(all_runs: dict) -> dict:
    """
    Compute mean ± std across seeds for each (task, method, condition, metric).
    Returns nested dict: task → method → condition → metric → {mean, std}
    """
    summary = {}
    for task, seeds_data in all_runs.items():
        summary[task] = {}
        # seeds_data: {seed_int: {method: {condition: {metric: float}}}}
        methods    = list(next(iter(seeds_data.values())).keys())
        conditions = ["iid", "shift"]
        metrics    = list(next(iter(next(iter(seeds_data.values())).values()))["iid"].keys())

        for method in methods:
            summary[task][method] = {}
            for cond in conditions:
                summary[task][method][cond] = {}
                for metric in metrics:
                    vals = [
                        seeds_data[s][method][cond][metric]
                        for s in seeds_data
                        if metric in seeds_data[s][method][cond]
                    ]
                    if vals:
                        summary[task][method][cond][metric] = {
                            "mean": float(np.mean(vals)),
                            "std":  float(np.std(vals, ddof=1)),
                        }
    return summary


def run_stats_tests(all_runs: dict, metric: str = "ece") -> dict:
    """
    Run pairwise Wilcoxon tests across seeds for a given metric.
    Returns nested dict: task → condition → pair → test results.
    """
    stats_out = {}
    for task, seeds_data in all_runs.items():
        stats_out[task] = {}
        methods    = list(next(iter(seeds_data.values())).keys())
        for cond in ["iid", "shift"]:
            metric_dict = {
                m: [seeds_data[s][m][cond].get(metric, np.nan)
                    for s in seeds_data]
                for m in methods
            }
            results = wilcoxon_pairwise(metric_dict)
            stats_out[task][cond] = {
                str(k): {
                    kk: bool(vv) if hasattr(vv, 'item') else float(vv) if isinstance(vv, float) else vv
                    for kk, vv in v.items()
                }
                for k, v in results.items()
            }
    return stats_out


def main():
    all_runs = {}

    for task in TASKS:
        csv_path = os.path.join(PROC_DIR, f"{task}_features.csv")
        if not os.path.exists(csv_path):
            print(f"[SKIP] {csv_path} not found. Run data/preprocess.py first.")
            continue

        print(f"\n{'='*60}")
        print(f"  Task: {task}")
        print(f"{'='*60}")
        all_runs[task] = {}

        for seed in SEEDS:
            print(f"\n  Seed {seed}:")
            t0 = time.time()
            all_runs[task][seed] = run_one_seed(task, seed)
            print(f"  Seed {seed} done in {time.time()-t0:.1f}s")

    # ── Save raw results ──────────────────────────────────────────────────────
    raw_path = os.path.join(RESULTS_DIR, "results_raw.json")
    with open(raw_path, "w") as f:
        # Convert int keys to str for JSON
        json.dump({t: {str(s): v for s, v in sv.items()}
                   for t, sv in all_runs.items()}, f, indent=2)
    print(f"\nRaw results → {raw_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = aggregate_results(all_runs)
    sum_path = os.path.join(RESULTS_DIR, "results_summary.json")
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary     → {sum_path}")

    # ── Stats tests ───────────────────────────────────────────────────────────
    for metric in ["ece", "brier", "auroc"]:
        st = run_stats_tests(all_runs, metric)
        st_path = os.path.join(RESULTS_DIR, f"stats_{metric}.json")
        with open(st_path, "w") as f:
            json.dump(st, f, indent=2)
        print(f"Stats ({metric}) → {st_path}")

    # ── Print summary table ───────────────────────────────────────────────────
    print_summary_table(summary)


def print_summary_table(summary: dict):
    """Quick console preview of main results."""
    print("\n\n" + "="*70)
    print("RESULTS SUMMARY (mean ± std)")
    print("="*70)
    key_metrics = ["auroc", "ece", "brier"]
    for task in summary:
        print(f"\nTask: {task}")
        header = f"{'Method':<15}" + "".join(
            f"{'IID '+m:>15}{'SHF '+m:>15}" for m in key_metrics)
        print(header)
        for method in summary[task]:
            row = f"{method:<15}"
            for m in key_metrics:
                for cond in ["iid", "shift"]:
                    d = summary[task][method][cond].get(m, {})
                    mn, sd = d.get("mean", np.nan), d.get("std", np.nan)
                    row += f"{mn:7.3f}±{sd:.3f}".rjust(15)
            print(row)


if __name__ == "__main__":
    main()
