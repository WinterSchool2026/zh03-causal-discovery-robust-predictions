# Causal Discovery and Causal Feature Selection for Robust Prediction

This repository contains a series of tutorial notebooks exploring the intersection of causal inference and machine learning. The central question: **can we do better than standard feature selection when we care about robustness under distribution shift?**

**Prerequisites:** basic probability, linear regression, familiarity with Python/sklearn.  
**Packages:** `numpy`, `sklearn`, `causallearn`, `cdt`, `dcor`, and custom helpers (`generate_scm`, `MB`, `causal_discovery`, `causal_feature_selection`).

---

## Notebooks

### 01 — Pairwise Causal Discovery

[![Open 01 – Pairwise Causal Discovery in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/homerdurand/zh03-causal-discovery-robust-predictions/blob/main/01_pairwise_causal_discovery.ipynb)

Given two variables X and Y, can we determine which one causes the other from observational data alone? This notebook introduces **Structural Causal Models (SCMs)** and the **RESIT algorithm** (Regression with Subsequent Independence Test), which exploits the asymmetry of cause-and-effect noise to identify causal direction. Covers linear and nonlinear settings using KCI (Kernel-based Conditional Independence) tests, and benchmarks against the Tübingen cause-effect pairs dataset.

---

### 02 — Markov Equivalence Classes

[![Open 02 – Markov Equivalence Classes in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/homerdurand/zh03-causal-discovery-robust-predictions/blob/main/02_markov_equivalence_classes.ipynb)

Before scaling up to multivariate graphs, this notebook addresses a fundamental limit: **observational data alone cannot always distinguish between all DAGs in the same Markov equivalence class**. Covers v-structures (colliders, mediators, confounders), CPDAGs, and what partial correlation can and cannot reveal. Essential background for understanding what constraint-based algorithms like PC and FCI can provably recover.

---

### 03 — Multivariate Causal Discovery

[![Open 03 – Multivariate Causal Discovery in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/homerdurand/zh03-causal-discovery-robust-predictions/blob/main/03_multivariate_causal_discovery.ipynb)

Extends pairwise ideas to full graph recovery over many variables. Introduces three families of algorithms — **constraint-based** (PC, FCI), **score-based** (GES), and **functional** (LiNGAM) — and compares their assumptions, outputs, and failure modes on synthetic SCMs. Discusses causal sufficiency, faithfulness, and the difference between DAGs and CPDAGs (equivalence class outputs).

---

### 04 — Multivariate Causal Feature Selection

[![Open 04 – Multivariate Causal Feature Selection in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/homerdurand/zh03-causal-discovery-robust-predictions/blob/main/04_causal_feature_selection.ipynb)

Rather than recovering the full graph, this notebook focuses on identifying the **Markov Blanket (MB)** of a target variable Y — the minimal sufficient feature set for prediction. Covers MB-discovery algorithms (HITON-MB, IAMB, MMPC, MB-GES, RESIT-MB) and evaluates their accuracy on synthetic data with varying sample sizes and graph structures.

---

### 05 — CFS for Robust Prediction

[![Open 05 – CFS for Robust Prediction in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/homerdurand/zh03-causal-discovery-robust-predictions/blob/main/05_cfs_robust_prediction.ipynb)

Demonstrates why **causal parents** of Y are more stable predictors than correlated features under distribution shift. Simulates intervention shifts on a synthetic SCM and compares R² degradation across feature selectors: correlation-based, MB-based, and parent-based. Shows that features selected purely for predictive power in training can collapse under interventions, while parent-based features remain stable.

---

### 06 — Invariant Causal Predictors (ICP)

[![Open 06 – Invariant Causal Predictors in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/homerdurand/zh03-causal-discovery-robust-predictions/blob/main/06_invariant_causal_predictors.ipynb)

Introduces **Invariant Causal Prediction** (Peters, Bühlmann & Meinshausen, 2016): a multi-environment method that identifies causal parents of Y directly from invariance of the conditional distribution across environments — without requiring full graph recovery. Covers the ICP algorithm, environment construction, invariance testing, and a head-to-head comparison with other selectors under intervention shifts. Discusses scaling limitations (exponential in the number of features) and practical workarounds.

---

## Summary

| # | Notebook | Core method | Key concept |
|---|----------|-------------|-------------|
| 1 | Pairwise Causal Discovery | RESIT | Noise asymmetry |
| 2 | Markov Equivalence Classes | CPDAG | Identifiability limits |
| 3 | Multivariate Causal Discovery | PC, GES, FCI, LiNGAM | Full graph recovery |
| 4 | Causal Feature Selection | HITON-MB, IAMB, MB-GES | Markov Blanket |
| 5 | CFS for Robust Prediction | Parent-based selection | Distribution shift |
| 6 | Invariant Causal Predictors | ICP | Multi-environment invariance |