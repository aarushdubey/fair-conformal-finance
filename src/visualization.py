import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List, Any

class ConformalVisualizer:
    """Professional visualization suite for Conformal Prediction metrics."""
    
    def __init__(self, style: str = "seaborn-v0_8-whitegrid"):
        try:
            plt.style.use(style)
        except:
            plt.style.use("default")
            
        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "figure.titlesize": 14,
        })

    def plot_coverage_vs_disparity(
        self, 
        results_summary: Dict[str, Dict[str, str]], 
        output_path: str
    ):
        """Plot the Pareto frontier of coverage vs set-size disparity."""
        plt.figure(figsize=(8, 6), dpi=300)
        
        # Colors for methods
        colors = {
            "Standard CP": "#94A3B8",
            "Mondrian CP": "#EF4444",
            "LC-CP": "#F59E0B",
            "Generic Fair CP": "#10B981",
            "FairTransCP (Ours)": "#2563EB"
        }
        
        for method, metrics in results_summary.items():
            try:
                cov = float(metrics["worst_group_coverage"].split(" +/-")[0])
                disp = float(metrics["set_size_disparity"].split(" +/-")[0])
                plt.scatter(cov, disp, label=method, color=colors.get(method, "black"), s=100)
            except (KeyError, ValueError):
                continue
                
        plt.xlabel("Worst-Group Coverage Rate", fontweight="bold")
        plt.ylabel("Set-Size Disparity Ratio", fontweight="bold")
        plt.title("Coverage-Equity Trade-off", fontweight="bold")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path)
        plt.close()

    def plot_group_coverage(
        self, 
        group_coverage: Dict[str, float], 
        group_name: str, 
        output_path: str
    ):
        """Plot bar chart of coverage per group."""
        plt.figure(figsize=(8, 5), dpi=300)
        groups = list(group_coverage.keys())
        values = list(group_coverage.values())
        
        plt.bar(groups, values, color="#2563EB", alpha=0.8, edgecolor="#1D4ED8")
        plt.axhline(0.9, color="red", linestyle="--", label="Target (0.9)")
        plt.ylabel("Coverage Rate", fontweight="bold")
        plt.title(f"Group Coverage: {group_name}", fontweight="bold")
        plt.legend()
        plt.savefig(output_path)
        plt.close()
