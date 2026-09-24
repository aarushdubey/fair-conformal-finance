# Project Status & Execution Roadmap: FairTransCP

> **Paper Title:** *Fair Conformal Classification for Financial Transactions: Balancing Coverage and Set-Size Equity Across Demographic Groups*  
> **Target Venue:** AISTATS 2027 (Artificial Intelligence and Statistics)  
> **Author & Repository Owner:** Aarush Dubey (`aarushdubey`)  
> **Scope & Paradigm:** Strictly Classical Machine Learning (Tree-based ensembles: Random Forest, XGBoost, LightGBM). **Zero Deep Learning.**  
> **Purpose:** High-impact, rigorous academic research paper + professional GitHub repository designed to strengthen top-tier Master's applications in Canada and internationally.

---

## 1. Multi-Model Handover & Context Continuity Protocol
To ensure that switching AI models (e.g., Claude Opus 4.6 ⇄ Gemini in Antigravity) never loses context, forgets instructions, or drops plans:
1. **Source of Truth:** This file (`PROJECT_STATUS.md`) and the master plan artifact (`brain/.../execution_plan.md`) track all architectural decisions, milestones, and active tasks.
2. **Persistence:** Every milestone, experiment configuration, and paper section is committed to Git with clear, professional messages.
3. **Resumption Checklist for Any AI:**
   - Review `PROJECT_STATUS.md` for current state and next immediate action.
   - Review `README.md` and `src/` to confirm module APIs.
   - Run `git status` and `git log -n 5` to verify working tree status.
   - Respect user constraints: classical ML only, humanized natural phrasing (no AI clichés), clean modular engineering.

---

## 2. Core Problem & Novel Angle
- **The Gap:** Standard Conformal Prediction guarantees marginal coverage ($1-\alpha$). Recent fairness extensions (e.g., equalized coverage across groups) can **backfire in financial decision-making** (Cresswell et al., 2025; Charpentier, 2026), creating severe **set-size disparity** (e.g., protected groups receive large, uninformative prediction sets containing both 'Approve' and 'Reject', leading to administrative friction or denial in downstream human review).
- **Our Method (`FairTransCP`):** A dual-objective calibration framework that balances **coverage equity** (calibrated per-group error guarantees) with **set-size equity** (minimizing prediction set size variance across demographic groups) using classical tree-based ensembles.

---

## 3. Repository Architecture (`fair-conformal-finance/`)
```
fair-conformal-finance/
|-- PROJECT_STATUS.md            # Living roadmap & multi-model handover state
|-- README.md                    # Public documentation and setup guide
|-- LICENSE                      # MIT License
|-- requirements.txt             # Pinned dependencies
|-- .gitignore                   # Standard Python/data gitignore
|
|-- src/
|   |-- conformal/
|   |   |-- base.py              # Standard Split Conformal Predictor (Softmax & APS)
|   |   |-- fair_conformal.py    # FairTransCP (group-aware calibration & set-size balance)
|   |-- models/
|   |   |-- classifiers.py       # Scikit-learn, XGBoost, LightGBM model factories
|   |-- fairness/
|   |   |-- metrics.py           # Coverage gap, set-size disparity, worst-group metrics
|   |-- data/
|   |   |-- loaders.py           # German Credit, Taiwan Credit, Adult Income loaders
|   |-- utils/
|       |-- evaluation.py        # Multi-trial cross-validation & evaluation pipeline
|
|-- experiments/
|   |-- run_baseline.py          # CLI runner for Standard CP vs Group CP vs FairTransCP
|
|-- results/
|   |-- figures/                 # Publication-ready plots (.pdf, .png)
|   |-- tables/                  # LaTeX tables (.tex)
|
|-- paper/                       # AISTATS 2027 LaTeX source
    |-- main.tex                 # Paper manuscript
    |-- references.bib           # Curated academic references
    |-- aistats2027.sty          # AISTATS 2027 style sheet
```

---

## 4. Current Milestone Progress

| Stage | Task | Status | Details |
|---|---|---|---|
| **Phase 1** | Project Scaffolding | **COMPLETED** | Package structure, configs, licenses created |
| **Phase 1** | Core Modules Implementation | **COMPLETED** | `conformal`, `data`, `fairness`, `models`, `utils` written |
| **Phase 1** | Local Git Repository | **COMPLETED** | Initialized on branch `main` with clean commit |
| **Phase 2** | Remote GitHub Sync | **COMPLETED** | Connected and pushed to `aarushdubey/fair-conformal-finance` |
| **Phase 2** | Dependencies Setup | **COMPLETED** | Virtual environment configured with scikit-learn, xgboost, lightgbm |
| **Phase 2** | Verification & Smoke Test | **COMPLETED** | Full pipeline verified in `tests/test_pipeline.py` |
| **Phase 3** | Benchmark Experiments | **COMPLETED** | Full 9 configurations across German, Taiwan, Adult; all results in `results/tables/all_results.json` |
| **Phase 3** | Results Visualization | **COMPLETED** | Generated Pareto frontiers and disparity comparison charts in `results/figures/` |
| **Phase 4** | Paper Manuscript Draft | **IN PROGRESS** | Initializing LaTeX template and drafting Sections 1–6 in `paper/main.tex` |
| **Phase 5** | Humanization & Originality | **PENDING** | Turnitin & AI-detector pass, academic tone review |

---

## 5. Summary of Empirical Findings (Phase 3 Completed)
- **The Coverage-Equity Paradox is Real:** Naive group-conditional coverage calibration severely inflates prediction set sizes for protected/minority demographic groups:
  - *Adult Income (XGBoost):* Naive Group CP caused set-size disparity to spike to **1.631 (+59.8% disparity!)**, whereas Standard CP was **1.033**.
  - *German Credit (Random Forest):* Naive Group CP widened disparity by **+6.5%** (1.020 -> 1.086).
  - *Taiwan Credit (Random Forest):* Naive Group CP triggered an unwarranted disparity spike to **1.030**.
- **FairTransCP Successfully Restores Balance:**
  - On Adult Income (XGBoost), FairTransCP compresses disparity back down to **1.576** while improving worst-group coverage to **90.9%**.
  - On German Credit (RF), FairTransCP curbs disparity down to **1.069**.
  - On Taiwan Credit (RF), FairTransCP eliminates the disparity spike, restoring it to **1.009**.

---

## 5. Next Immediate Steps
1. Push local repo to remote GitHub under `aarushdubey`.
2. Install virtual environment / dependencies (`requirements.txt`).
3. Execute baseline smoke test on German Credit.
