# FairTransCP: Fair Conformal Classification for Financial Transactions

> Balancing Coverage and Set-Size Equity Across Demographic Groups

This repository contains the code and experiments accompanying the paper:

**"Fair Conformal Classification for Financial Transactions: Balancing Coverage and Set-Size Equity Across Demographic Groups"**
*Submitted to AISTATS 2027*

---

## Overview

Financial machine learning models are increasingly used for high-stakes decisions like credit scoring and fraud detection. While **conformal prediction (CP)** offers distribution-free uncertainty quantification, standard CP methods ignore fairness: they may produce prediction sets that systematically disadvantage certain demographic groups.

Recent work has shown that naively enforcing **equalized coverage** (equal coverage rates across groups) can paradoxically *increase* disparate impact in downstream decisions — a phenomenon we term the **coverage–equity paradox**.

**FairTransCP** addresses this by jointly optimizing two fairness objectives:
1. **Coverage equity**: similar coverage rates across demographic groups
2. **Set-size equity**: similar prediction set sizes, so no group faces disproportionate ambiguity

We demonstrate our method on four financial datasets using purely classical ML models (Random Forest, XGBoost, LightGBM) — no deep learning required.

## Key Contributions

- A conformal prediction framework that **jointly calibrates** for coverage and set-size fairness
- **Theoretical guarantees** on worst-group coverage under fairness constraints
- Empirical evidence of the **coverage–equity paradox** in financial classification
- Extensive evaluation on **4 financial/credit datasets** with 3 classical ML base models

## Installation

```bash
# Clone the repo
git clone https://github.com/aarushdubey/fair-conformal-finance.git
cd fair-conformal-finance

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
fair-conformal-finance/
├── src/                     # Source code
│   ├── conformal/           # Conformal prediction core
│   ├── models/              # Base classifier wrappers
│   ├── fairness/            # Fairness metrics and constraints
│   ├── data/                # Data loading and preprocessing
│   └── utils/               # Plotting, evaluation, reproducibility
├── experiments/             # Experiment scripts and configs
├── notebooks/               # Exploratory analysis
├── results/                 # Generated figures and tables
├── paper/                   # LaTeX source for the paper
├── tests/                   # Unit tests
└── docs/                    # Additional documentation
```

## Quick Start

```bash
# Run baseline conformal prediction
python experiments/run_baseline.py --config experiments/configs/german_credit.yaml

# Run FairTransCP
python experiments/run_fair_conformal.py --config experiments/configs/german_credit.yaml

# Run ablation studies
python experiments/run_ablation.py --config experiments/configs/german_credit.yaml
```

## Datasets

| Dataset | Task | Samples | Features | Sensitive Attributes |
|---------|------|---------|----------|---------------------|
| [German Credit](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) | Credit risk | 1,000 | 20 | Age, Gender |
| [Taiwan Credit](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) | Default prediction | 30,000 | 23 | Gender, Education |
| [IEEE-CIS Fraud](https://www.kaggle.com/c/ieee-fraud-detection) | Fraud detection | 590,540 | 434 | Card type |
| [Adult Income](https://archive.ics.uci.edu/dataset/2/adult) | Income prediction | 48,842 | 14 | Race, Gender |

Datasets are downloaded automatically on first run.

## Reproducing Results

To reproduce all results from the paper:

```bash
# Full experiment suite (takes ~2 hours on a standard machine)
python experiments/run_baseline.py --all
python experiments/run_fair_conformal.py --all
python experiments/run_ablation.py --all
```

Results are saved in `results/tables/` and `results/figures/`.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{dubey2027fairtranscp,
  title={Fair Conformal Classification for Financial Transactions:
         Balancing Coverage and Set-Size Equity Across Demographic Groups},
  author={Dubey, Aarush},
  booktitle={Proceedings of the 30th International Conference on
             Artificial Intelligence and Statistics (AISTATS)},
  year={2027}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
