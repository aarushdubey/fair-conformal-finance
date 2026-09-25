# FairTransCP: Fair Conformal Classification for Financial Transactions

> Balancing Coverage and Set-Size Equity Across Demographic Groups

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aarushdubey/fair-conformal-finance/blob/main/notebooks/demo.ipynb)
[![Paper Under Review](https://img.shields.io/badge/AISTATS-2027-blue.svg)](#)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> 🚀 **Test Without Local Setup:** Click the **Open in Colab** badge above to execute the entire interactive workflow in your browser via Google Colab with zero installation.

This repository contains the code and experiments accompanying the paper:

**"Fair Conformal Classification for Financial Transactions: Balancing Coverage and Set-Size Equity Across Demographic Groups"**  
*Submitted to AISTATS 2027*

---

## Overview

Financial machine learning models are increasingly used for high-stakes decisions like credit scoring and fraud detection. While **conformal prediction (CP)** offers distribution-free uncertainty quantification, standard CP methods ignore fairness: they may produce prediction sets that systematically disadvantage certain demographic groups.

Recent literature demonstrates that naively enforcing **equalized coverage** (equal coverage rates across groups) can paradoxically *increase* disparate impact in downstream human-in-the-loop decisions — a phenomenon characterized as the **coverage–equity paradox**.

**FairTransCP** addresses this by jointly calibrating two fairness objectives:
1. **Coverage equity**: bounding coverage gaps across demographic groups
2. **Set-size equity**: minimizing prediction set-size disparity so protected groups do not face disproportionate ambiguity

We evaluate our method across 3 real-world financial/credit benchmarks using strictly classical ML ensembles (Random Forest, XGBoost, LightGBM) — fully explainable and compliant with financial regulations (FCRA, ECOA, EU AI Act).

## Key Contributions

- **Dual-Objective Calibration Framework**: Jointly optimizes coverage equity and set-size parity via regularized threshold interpolation.
- **Theoretical Lower Bound (Theorem 1)**: Formal proof bounding worst-group coverage loss under regularized conformal prediction.
- **SOTA Benchmarking**: Rigorous comparison against Standard CP, Mondrian CP (Vovk et al.), Label-Clustered CP (LC-CP), and Generic Fair CP.
- **Classical ML & Full Auditability**: Implemented exclusively with tree-based ensembles (Random Forest, XGBoost, LightGBM).
- **100% Reproducible**: End-to-end scripts, unit tests, and interactive Jupyter / Google Colab notebook.

## Installation

### Option 1: Conda (Recommended)

```bash
git clone https://github.com/aarushdubey/fair-conformal-finance.git
cd fair-conformal-finance

conda env create -f conda_env.yml
conda activate fair-conformal
```

### Option 2: Pip & Virtual Environment

```bash
git clone https://github.com/aarushdubey/fair-conformal-finance.git
cd fair-conformal-finance

python -m venv venv
# Linux/macOS:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

pip install -r requirements.txt
```

## Project Structure

```
fair-conformal-finance/
├── src/                     # Source code
│   ├── conformal/           # Conformal core & SOTA baselines (Mondrian, LC-CP, GenericFair)
│   ├── models/              # Base classifiers (Random Forest, XGBoost, LightGBM)
│   ├── fairness/            # Coverage gap and set-size disparity metrics
│   ├── data/                # Loaders for German Credit, Taiwan Credit, Adult Census
│   └── utils/               # Evaluation pipelines and metric summarizers
├── experiments/             # Baseline runner, sensitivity analysis, and figure generation
├── notebooks/               # 1-click Google Colab demo
├── paper/                   # Complete AISTATS 2027 LaTeX manuscript & style files
├── results/                 # Publication figures and JSON benchmark tables
├── tests/                   # Smoke tests and pipeline verification
├── conda_env.yml            # Conda environment specification
└── requirements.txt         # Pip dependencies
```

## Quick Start

Run an end-to-end experiment on German Credit with Random Forest:

```bash
python experiments/run_baseline.py --dataset german_credit --model rf
```

Run the complete benchmark suite across all 9 configurations:

```bash
python experiments/run_baseline.py --all
```

Run sensitivity analysis across miscoverage rates $\alpha \in [0.01, 0.05, 0.1, 0.2]$:

```bash
python experiments/run_baseline.py --all --sensitivity
```

Generate all publication figures:

```bash
python experiments/plot_figures.py
```

Run unit tests:

```bash
python tests/test_pipeline.py
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
