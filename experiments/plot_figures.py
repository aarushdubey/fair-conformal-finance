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

    alpha_key = "alpha_0.1" # Use 0.1 as representative

    labels = []
    std_vals = []
    mon_vals = []
    lcc_vals = []
    gen_vals = []
    fair_vals = []

    for key, label in configs:
        if key in data and alpha_key in data[key]:
            res = data[key][alpha_key]
            labels.append(label)
            std_vals.append(parse_val(res.get("Standard CP", {}).get("set_size_disparity", "1.0")))
            mon_vals.append(parse_val(res.get("Mondrian CP", {}).get("set_size_disparity", "1.0")))
            lcc_vals.append(parse_val(res.get("LC-CP", {}).get("set_size_disparity", "1.0")))
            gen_vals.append(parse_val(res.get("Generic Fair CP", {}).get("set_size_disparity", "1.0")))
            fair_vals.append(parse_val(res.get("FairTransCP (Ours)", {}).get("set_size_disparity", "1.0")))

    x = np.arange(len(labels))
    width = 0.15

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    ax.bar(x - 2*width, std_vals, width, label="Standard CP", color="#94A3B8", edgecolor="#475569")
    ax.bar(x - width, mon_vals, width, label="Mondrian CP", color="#EF4444", edgecolor="#B91C1C")
    ax.bar(x, lcc_vals, width, label="LC-CP", color="#F59E0B", edgecolor="#B45309")
    ax.bar(x + width, gen_vals, width, label="Generic Fair CP", color="#10B981", edgecolor="#047857")
    ax.bar(x + 2*width, fair_vals, width, label="FairTransCP (Ours)", color="#2563EB", edgecolor="#1D4ED8")

    ax.set_ylabel("Set-Size Disparity Ratio (1.0 = Perfect Equity)", fontweight="bold")
    ax.set_title("Empirical Demonstration of the Coverage-Equity Trap in Financial Benchmarks", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7, label="Ideal Equity (1.0)")
    ax.legend(frameon=True, loc="upper left")

    # Add annotations for the disparity spike
    for i in range(len(labels)):
        diff = ((mon_vals[i] - std_vals[i]) / std_vals[i]) * 100
        if diff > 1.0:
            ax.annotate(
                f"+{diff:.1f}% disparity",
                xy=(x[i], mon_vals[i]),
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

    alpha_key = "alpha_0.1"

    for key, name in names.items():
        if key not in data or alpha_key not in data[key]:
            continue
        res = data[key][alpha_key]
        m = markers[key]

        # Standard CP
        cov_std = parse_val(res.get("Standard CP", {}).get("worst_group_coverage", "0.0"))
        disp_std = parse_val(res.get("Standard CP", {}).get("set_size_disparity", "1.0"))
        ax.scatter(cov_std, disp_std, color="#94A3B8", marker=m, s=70, label=f"Standard CP ({name})" if key == "german_credit_rf" else "")

        # Mondrian CP
        cov_mon = parse_val(res.get("Mondrian CP", {}).get("worst_group_coverage", "0.0"))
        disp_mon = parse_val(res.get("Mondrian CP", {}).get("set_size_disparity", "1.0"))
        ax.scatter(cov_mon, disp_mon, color="#EF4444", marker=m, s=70, label=f"Mondrian CP ({name})" if key == "german_credit_rf" else "")

        # LC-CP
        cov_lcc = parse_val(res.get("LC-CP", {}).get("worst_group_coverage", "0.0"))
        disp_lcc = parse_val(res.get("LC-CP", {}).get("set_size_disparity", "1.0"))
        ax.scatter(cov_lcc, disp_lcc, color="#F59E0B", marker=m, s=70, label=f"LC-CP ({name})" if key == "german_credit_rf" else "")

        # Generic Fair CP
        cov_gen = parse_val(res.get("Generic Fair CP", {}).get("worst_group_coverage", "0.0"))
        disp_gen = parse_val(res.get("Generic Fair CP", {}).get("set_size_disparity", "1.0"))
        ax.scatter(cov_gen, disp_gen, color="#10B981", marker=m, s=70, label=f"Generic Fair CP ({name})" if key == "german_credit_rf" else "")

        # FairTransCP
        cov_fair = parse_val(res.get("FairTransCP (Ours)", {}).get("worst_group_coverage", "0.0"))
        disp_fair = parse_val(res.get("FairTransCP (Ours)", {}).get("set_size_disparity", "1.0"))
        ax.scatter(cov_fair, disp_fair, color="#2563EB", marker=m, s=110, label=f"FairTransCP ({name})" if key == "german_credit_rf" else "")

        # Draw trajectory arrow from Mondrian CP to FairTransCP
        ax.annotate(
            "",
            xy=(cov_fair, disp_fair),
            xytext=(cov_mon, disp_mon),
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
