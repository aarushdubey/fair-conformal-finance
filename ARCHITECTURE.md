# Project Architecture: FairTransCP

FairTransCP is designed as a modular framework for researching fair conformal prediction.

## High-Level Flow
`Data Loaders` $\rightarrow$ `Base Classifiers` $\rightarrow$ `Conformal Calibration` $\rightarrow$ `Fairness Adjustment` $\rightarrow$ `Evaluation Metrics`

## Module Breakdown

### 1. `src.data`
Handles the ingestion of financial benchmarks. Each loader returns a standardized dictionary containing features, labels, and sensitive attributes.

### 2. `src.models`
Provides wrappers for classical tree-based ensembles. We prioritize models like Random Forest and XGBoost for their explainability and regulatory compliance in finance.

### 3. `src.conformal`
The core engine. 
- `base.py`: Implements the standard split conformal prediction logic.
- `baselines.py`: Implements SOTA baselines (Mondrian, LCCP, GenericFair).
- `fair_conformal.py`: Implements our la-weighted optimization for coverage and set-size equity.

### 4. `src.fairness`
Implements key fairness metrics, specifically focusing on the **Coverage Gap** and **Set-Size Disparity Ratio**.

### 5. `src.utils`
Contains the `evaluation` pipeline that orchestrates the training, calibration, and testing across multiple trials to ensure statistical significance.

## Design Principles
- **Reproducibility**: Everything is driven by a `config.yaml` and fixed random seeds.
- **Modularity**: New conformal methods can be added by inheriting from `SplitConformalClassifier`.
- **Transparency**: We avoid "black box" deep learning to ensure auditability for financial regulators.
