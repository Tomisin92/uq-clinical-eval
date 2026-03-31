"""
generate_latex_tables.py
------------------------
Reads results/results_summary.json and results/stats_*.json,
then writes LaTeX table code to results/latex_tables.tex

Run AFTER run_experiments.py.
"""

import os, json
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def fmt(mean, std, bold=False):
    s = f"{mean:.3f} $\\pm$ {std:.3f}"
    return f"\\textbf{{{s}}}" if bold else s


def load(fname):
    path = os.path.join(RESULTS_DIR, fname)
    with open(path) as f:
        return json.load(f)


def best_method(summary, task, cond, metric, lower_is_better=True):
    methods = list(summary[task].keys())
    vals = {m: summary[task][m][cond].get(metric, {}).get("mean", np.nan)
            for m in methods}
    if lower_is_better:
        return min(vals, key=lambda m: vals[m])
    else:
        return max(vals, key=lambda m: vals[m])


def table_iid(summary):
    """Table 1: In-distribution performance."""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{In-distribution performance on mortality and readmission "
                 r"prediction tasks. Values are mean $\pm$ SD across five independent runs. "
                 r"Bold indicates best result per metric. "
                 r"$\downarrow$ = lower is better; $\uparrow$ = higher is better.}")
    lines.append(r"\label{tab:iid}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llccccc}")
    lines.append(r"\toprule")
    lines.append(r"Task & Method & AUROC$\uparrow$ & ECE$\downarrow$ & "
                 r"ACE$\downarrow$ & Brier$\downarrow$ & AUPRC$\uparrow$ \\")
    lines.append(r"\midrule")

    method_labels = {
        "mc_dropout": "MC Dropout",
        "bnn":        "BNN",
        "conformal":  "Conformal Pred.",
    }

    for task_key, task_label in [("mortality", "Mortality"),
                                  ("readmission_30d", "Readmission")]:
        if task_key not in summary:
            continue
        first = True
        n_methods = len(summary[task_key])
        for method_key, method_label in method_labels.items():
            if method_key not in summary[task_key]:
                continue
            cond = "iid"
            data = summary[task_key][method_key][cond]

            row_cells = []
            for metric, low in [("auroc", False), ("ece", True),
                                 ("ace", True),   ("brier", True), ("auprc", False)]:
                d = data.get(metric, {})
                mn = d.get("mean", float("nan"))
                sd = d.get("std",  float("nan"))
                best = best_method(summary, task_key, cond, metric,
                                   lower_is_better=low)
                row_cells.append(fmt(mn, sd, bold=(method_key == best)))

            task_col = (r"\multirow{" + str(n_methods) + r"}{*}{" + task_label + "}"
                        if first else "")
            lines.append(f"  {task_col} & {method_label} & " +
                         " & ".join(row_cells) + r" \\")
            first = False

        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def table_shift(summary):
    """Table 2: Distribution shift results."""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Performance degradation under temporal distribution shift "
                 r"(shifted test set vs.\ i.i.d.\ test set). "
                 r"$\Delta$ values are shift $-$ i.i.d.; positive values indicate degradation "
                 r"for error metrics. Coverage is empirical marginal coverage; nominal = 0.90.}")
    lines.append(r"\label{tab:shift}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llccccc}")
    lines.append(r"\toprule")
    lines.append(r"Task & Method & $\Delta$ECE$\downarrow$ & $\Delta$Brier$\downarrow$ & "
                 r"$\Delta$AUROC$\uparrow$ & Coverage & Avg. Set Size \\")
    lines.append(r"\midrule")

    method_labels = {
        "mc_dropout": "MC Dropout",
        "bnn":        "BNN",
        "conformal":  "Conformal Pred.",
    }

    for task_key, task_label in [("mortality", "Mortality"),
                                  ("readmission_30d", "Readmission")]:
        if task_key not in summary:
            continue
        first = True
        n_methods = len(summary[task_key])
        for method_key, method_label in method_labels.items():
            if method_key not in summary[task_key]:
                continue

            def _delta(metric):
                iid_d   = summary[task_key][method_key]["iid"].get(metric, {})
                shift_d = summary[task_key][method_key]["shift"].get(metric, {})
                mn = shift_d.get("mean", np.nan) - iid_d.get("mean", np.nan)
                # propagate std approximately
                sd = np.sqrt(iid_d.get("std", 0)**2 + shift_d.get("std", 0)**2)
                return mn, sd

            def _raw(metric):
                d = summary[task_key][method_key]["shift"].get(metric, {})
                return d.get("mean", np.nan), d.get("std", np.nan)

            d_ece_mn,   d_ece_sd   = _delta("ece")
            d_brier_mn, d_brier_sd = _delta("brier")
            d_auc_mn,   d_auc_sd   = _delta("auroc")
            cov_mn,     cov_sd     = _raw("coverage")
            sz_mn,      sz_sd      = _raw("avg_set_size")

            cov_str = (fmt(cov_mn, cov_sd) if not np.isnan(cov_mn) else "N/A")
            sz_str  = (fmt(sz_mn,  sz_sd)  if not np.isnan(sz_mn)  else "N/A")

            task_col = (r"\multirow{" + str(n_methods) + r"}{*}{" + task_label + "}"
                        if first else "")
            lines.append(
                f"  {task_col} & {method_label} & "
                f"{fmt(d_ece_mn, d_ece_sd)} & "
                f"{fmt(d_brier_mn, d_brier_sd)} & "
                f"{fmt(d_auc_mn, d_auc_sd)} & "
                f"{cov_str} & "
                f"{sz_str} \\\\"
            )
            first = False

        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def table_stats(stats_ece):
    """Table 3: Pairwise statistical test results."""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Pairwise Wilcoxon signed-rank test results for ECE "
                 r"under temporal shift. $p$-values are Holm--Bonferroni corrected. "
                 r"$^{*}p < 0.05$; $^{**}p < 0.01$.}")
    lines.append(r"\label{tab:stats}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llcc}")
    lines.append(r"\toprule")
    lines.append(r"Task & Comparison & $p_{\text{adj}}$ & Significant \\")
    lines.append(r"\midrule")

    pair_labels = {
        "('mc_dropout', 'bnn')":        "MC Dropout vs.\\ BNN",
        "('mc_dropout', 'conformal')":  "MC Dropout vs.\\ Conformal",
        "('bnn', 'conformal')":         "BNN vs.\\ Conformal",
    }

    for task_key, task_label in [("mortality", "Mortality"),
                                  ("readmission_30d", "Readmission")]:
        if task_key not in stats_ece:
            continue
        data = stats_ece[task_key].get("shift", {})
        first = True
        for pair_key, pair_label in pair_labels.items():
            if pair_key not in data:
                continue
            p_adj = data[pair_key]["p_adj"]
            sig   = data[pair_key]["significant"]
            stars = ("$^{**}$" if p_adj < 0.01 else
                     "$^{*}$"  if p_adj < 0.05 else "n.s.")
            task_col = (r"\multirow{3}{*}{" + task_label + "}" if first else "")
            lines.append(f"  {task_col} & {pair_label} & "
                         f"{p_adj:.4f} & {stars} \\\\")
            first = False
        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    try:
        summary = load("results_summary.json")
    except FileNotFoundError:
        print("results_summary.json not found. Run run_experiments.py first.")
        return

    try:
        stats_ece = load("stats_ece.json")
    except FileNotFoundError:
        stats_ece = {}

    out_path = os.path.join(RESULTS_DIR, "latex_tables.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("% ── Auto-generated LaTeX tables ─────────────────────────────\n")
        f.write("% Include this file in your paper with \\input{results/latex_tables}\n\n")
        f.write(table_iid(summary))
        f.write("\n\n")
        f.write(table_shift(summary))
        f.write("\n\n")
        if stats_ece:
            f.write(table_stats(stats_ece))
            f.write("\n")

    print(f"LaTeX tables → {out_path}")


if __name__ == "__main__":
    main()
