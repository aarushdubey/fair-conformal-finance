"""
Generate publication-quality figures for the AISTATS 2027 paper.

Plots:
1. Coverage Gap vs. Set-Size Disparity (Pareto trade-off)
2. Disparity Spike Comparison across datasets (Standard CP vs Group CP vs FairTransCP)
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Set style for academic publication
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
})

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def parse_val(str_val):
    """Extract float mean from 'mean +/- std' string."""
    try:
        return float(str_val.split("+/-")[0].strip())
    except Exception:
        return 0.0


def plot_disparity_comparison():
    """Bar chart showing set-size disparity across methods on key benchmarks."""
    json_path = os.path.join(TABLES_DIR, "all_results.json")
    if not os.path.exists(json_path):
        print("all_results.json not found, skipping plot.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    # Key configs where disparity trade-off is prominent
    configs = [
        ("german_credit_rf", "German Credit (RF)"),
        ("taiwan_credit_rf", "Taiwan Credit (RF)"),
        ("adult_income_rf", "Adult Income (RF)"),
        ("adult_income_xgboost", "Adult Income (XGBoost)"),
    ]

    labels = []
    std_vals = []
    grp_vals = []
    fair_vals = []

    for key, label in configs:
        if key in data:
            labels.append(label)
            std_vals.append(parse_val(data[key]["Standard CP"]["set_size_disparity"]))
            grp_vals.append(parse_val(data[key]["Group-Conditional CP"]["set_size_disparity"]))
            fair_vals.append(parse_val(data[key]["FairTransCP (Ours)"]["set_size_disparity"]))

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    rects1 = ax.bar(x - width, std_vals, width, label="Standard CP (Marginal)", color="#94A3B8", edgecolor="#475569")
    rects2 = ax.bar(x, grp_vals, width, label="Group-Conditional CP (Equalized)", color="#EF4444", edgecolor="#B91C1C")
    rects3 = ax.bar(x + width, fair_vals, width, label="FairTransCP (Ours: Balanced)", color="#2563EB", edgecolor="#1D4ED8")

    ax.set_ylabel("Set-Size Disparity Ratio (1.0 = Perfect Equity)", fontweight="bold")
    ax.set_title("Empirical Demonstration of the Coverage-Equity Trap in Financial Benchmarks", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7, label="Ideal Equity (1.0)")
    ax.legend(frameon=True, loc="upper left")

    # Add annotations for the disparity spike
    for i in range(len(labels)):
        diff = ((grp_vals[i] - std_vals[i]) / std_vals[i]) * 100
        if diff > 1.0:
            ax.annotate(
                f"+{diff:.1f}% disparity",
                xy=(x[i], grp_vals[i]),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                fontweight="bold",
                color="#B91C1C"
            )

    plt.tight_layout()
    out_png = os.path.join(FIGURES_DIR, "set_size_disparity_comparison.png")
    out_pdf = os.path.join(FIGURES_DIR, "set_size_disparity_comparison.pdf")
    plt.savefig(out_png)
    plt.savefig(out_pdf)
    plt.close()
    print(f"Generated: {out_png} and {out_pdf}")


def plot_pareto_frontier():
    """Scatter plot showing Worst-Group Coverage vs. Set-Size Disparity."""
    json_path = os.path.join(TABLES_DIR, "all_results.json")
    if not os.path.exists(json_path):
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    markers = {"german_credit_rf": "o", "taiwan_credit_rf": "s", "adult_income_rf": "^", "adult_income_xgboost": "D"}
    names = {
        "german_credit_rf": "German (RF)",
        "taiwan_credit_rf": "Taiwan (RF)",
        "adult_income_rf": "Adult (RF)",
        "adult_income_xgboost": "Adult (XGB)",
    }

    for key, name in names.items():
        if key not in data:
            continue
        m = markers[key]
        
        # Standard CP
        cov_std = parse_val(data[key]["Standard CP"]["worst_group_coverage"])
        disp_std = parse_val(data[key]["Standard CP"]["set_size_disparity"])
        ax.scatter(cov_std, disp_std, color="#94A3B8", marker=m, s=90, label=f"Standard CP ({name})" if key == "german_credit_rf" else "")

        # Group CP
        cov_grp = parse_val(data[key]["Group-Conditional CP"]["worst_group_coverage"])
        disp_grp = parse_val(data[key]["Group-Conditional CP"]["set_size_disparity"])
        ax.scatter(cov_grp, disp_grp, color="#EF4444", marker=m, s=90, label=f"Group CP ({name})" if key == "german_credit_rf" else "")

        # FairTransCP
        cov_fair = parse_val(data[key]["FairTransCP (Ours)"]["worst_group_coverage"])
        disp_fair = parse_val(data[key]["FairTransCP (Ours)"]["set_size_disparity"])
        ax.scatter(cov_fair, disp_fair, color="#2563EB", marker=m, s=110, label=f"FairTransCP ({name})" if key == "german_credit_rf" else "")

        # Draw trajectory arrow from Group CP to FairTransCP
        ax.annotate(
            "",
            xy=(cov_fair, disp_fair),
            xytext=(cov_grp, disp_grp),
            arrowprops=dict(arrowstyle="->", color="#2563EB", lw=1.2, ls="--"),
        )
        ax.text(cov_fair + 0.001, disp_fair, name, fontsize=8, color="#1E293B")

    ax.set_xlabel("Worst-Group Coverage Rate (Higher is Safer, Target = 0.90)", fontweight="bold")
    ax.set_ylabel("Set-Size Disparity Ratio (Lower is Fairer)", fontweight="bold")
    ax.set_title("Pareto Balance: FairTransCP Mitigates Disparity Inflation", fontweight="bold", pad=12)
    ax.axvline(0.90, color="green", linestyle=":", label="90% Nominal Coverage")

    plt.tight_layout()
    out_png = os.path.join(FIGURES_DIR, "pareto_coverage_vs_disparity.png")
    out_pdf = os.path.join(FIGURES_DIR, "pareto_coverage_vs_disparity.pdf")
    plt.savefig(out_png)
    plt.savefig(out_pdf)
    plt.close()
    print(f"Generated: {out_png} and {out_pdf}")


if __name__ == "__main__":
    plot_disparity_comparison()
    plot_pareto_frontier()
