# Data Imputation, Imbalanced Classification & Explainability

> A three-notebook pipeline that converts a financial regression dataset into a 5-class ordinal problem, benchmarks four missing-data imputation methods, evaluates six imbalance-correction strategies, and explains the winning classifier using SHAP TreeExplainer — all on the same NASDAQ-100 dataset from Assignment 1.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Dataset & Feature Engineering](#3-dataset--feature-engineering)
4. [Problem Framing: From Regression to Classification](#4-problem-framing-from-regression-to-classification)
5. [Notebook 01 — EDA & Target Creation](#5-notebook-01--eda--target-creation)
6. [Notebook 02 — Imputation Methods](#6-notebook-02--imputation-methods)
7. [Notebook 03 — Imbalanced Classification & Explainability](#7-notebook-03--imbalanced-classification--explainability)
8. [Key Results Summary](#8-key-results-summary)
9. [Known Issues & Discrepancies](#9-known-issues--discrepancies)
10. [How to Run](#10-how-to-run)
11. [Dependencies](#11-dependencies)

---

## 1. Project Overview

Assignment 2 reuses the financial dataset from Assignment 1 (NASDAQ-100 and related assets, 2010–2024) but introduces three new technical challenges not present in a standard regression workflow:

1. **Missing data** — 5% MCAR (Missing Completely At Random) missingness is injected into 8 technical indicator columns, and four imputation methods are benchmarked on both reconstruction quality and downstream classification performance.
2. **Class imbalance** — the classification target is heavily skewed (Neutral ≈ 52%, Strong Drop ≈ 6%), requiring explicit imbalance-correction strategies evaluated under Macro F1 rather than accuracy.
3. **Explainability** — SHAP (SHapley Additive exPlanations) TreeExplainer is applied to the winning classifier to identify which features drive predictions for each of the 5 classes.

The pipeline is structured as three sequential notebooks, each backed by a modular `utils/` library that separates data logic from notebook presentation.

---

## 2. Repository Structure

```
Assignment2/
├── notebooks/
│   ├── 01_EDA_and_Target_Creation.ipynb           # EDA, class distribution analysis
│   ├── 02_Imputation_Methods.ipynb                # 4-method imputation benchmark
│   └── 03_Imbalanced_Classification_&_Explainability.ipynb  # Classification + SHAP
├── utils/
│   ├── data_loader.py      # Preprocessing, feature engineering, target creation, MCAR injection
│   ├── imputation.py       # Imputer catalogue, reconstruction quality, downstream F1
│   ├── imbalance.py        # Resampling strategy catalogue and evaluation
│   └── evaluation.py       # Plotting utilities (bar charts, confusion matrices, heatmaps)
├── data/
│   ├── financial_regression.csv    # Raw input (same as Assignment 1)
│   ├── df_clean.parquet            # Preprocessed dataset, no artificial missingness (ground truth)
│   └── df_missing.parquet          # Preprocessed dataset with 5% MCAR injected into 8 columns
├── outputs/
│   ├── figures/                    # All saved plots (29 PNG files)
│   ├── models/
│   │   └── best_pipeline.joblib    # Persisted best-tuned sklearn/imblearn pipeline
│   └── reports/
│       ├── 02_imputation_reconstruction_quality.csv
│       ├── 03_strategy_comparison.csv
│       └── performance_summary.csv
├── 01_EDA.ipynb                    # Root-level EDA notebook
├── VisualAnalytics.ipynb           # Visual analytics companion
└── Project_Memorie.md              # Full technical documentation
```

---

## 3. Dataset & Feature Engineering

**Source:** Same `financial_regression.csv` as Assignment 1 — [Kaggle Financial Data](https://www.kaggle.com/datasets/franciscogcc/financial-data)

### Preprocessing Pipeline (identical to Assignment 1)

1. Load raw CSV, normalize column names, sort by date.
2. Forward-fill macro variables (GDP quarterly, CPI monthly, US rates) — last-known-value semantics.
3. Drop market holidays (rows where NASDAQ and SP500 close prices are both NaN).
4. Compute log returns: $r_t = \log(P_t / P_{t-1})$ for all 7 close-price columns.

### Technical Indicators (NASDAQ-focused, new in Assignment 2)

| Feature | Formula / Window | Role |
|---|---|---|
| `nasdaq_rsi_14` | RSI(14), Wilder EMA ($\alpha = 1/14$) | Momentum oscillator 0–100 |
| `nasdaq_macd` | EMA(12) − EMA(26) | Trend/momentum line |
| `nasdaq_macd_signal` | EMA(9) of MACD | Signal smoothing |
| `nasdaq_macd_hist` | MACD − signal | Momentum acceleration |
| `nasdaq_bb_upper/lower` | 20-day rolling mean ± 2 std | Bollinger Band boundaries |
| `nasdaq_bb_width` | (upper − lower) / mid | Relative band width ≈ local volatility |
| `nasdaq_lr_lag1/2/3/5` | `nasdaq_log_return.shift(k)` | Own-return autocorrelation features |
| `nasdaq_vol_5d` / `_20d` | rolling(5/20).std() on log return | Realized short/medium volatility |
| `sp500_lr_lag1`, `gold_lr_lag1` | `{asset}_log_return.shift(1)` | Cross-asset lagged returns |

### MCAR Missingness Injection

5% of values are erased independently at random per column in 8 technical indicator columns:

```
nasdaq_rsi_14, nasdaq_macd, nasdaq_macd_signal, nasdaq_macd_hist,
nasdaq_bb_width, nasdaq_lr_lag1, nasdaq_lr_lag2, nasdaq_vol_5d
```

MCAR (Missing Completely At Random) is chosen as the experimental design because it:
- guarantees unbiased imputation is theoretically possible,
- provides a known ground truth for evaluating reconstruction quality,
- represents the cleanest comparison scenario.

`df_clean.parquet` (ground truth) and `df_missing.parquet` (5% MCAR applied) are saved separately so reconstruction MAE can be measured at the exact positions that were erased.

### Chronological Split

```python
train = first 80% of rows by date
test  = last  20% of rows by date
```

No shuffling is ever applied. Rolling-window features and lagged targets make any random split a source of temporal leakage.

---

## 4. Problem Framing: From Regression to Classification

The next-day NASDAQ log return $r_{t+1}$ is converted into a **5-class ordinal label** using thresholds based on mean-and-sigma boundaries:

$$t_1 = \mu - 1.5\sigma, \quad t_2 = \mu - 0.5\sigma, \quad t_3 = \mu + 0.5\sigma, \quad t_4 = \mu + 1.5\sigma$$

| Class | Label | Return Range | Approx. Share | Imbalance vs Neutral |
|---|---|---|---|---|
| 0 | Strong Drop | $r < t_1$ | ≈6.3% | 8.1:1 |
| 1 | Mild Drop | $t_1 \le r < t_2$ | ≈16.5% | 3.1:1 |
| 2 | Neutral | $t_2 \le r \le t_3$ | ≈51.7% | — |
| 3 | Mild Rise | $t_3 < r \le t_4$ | ≈20.7% | 2.5:1 |
| 4 | Strong Rise | $r > t_4$ | ≈4.9% | 10.4:1 |

**Leakage prevention:** thresholds are computed on **training-split returns only** and applied verbatim to the test split. Labels never depend on future return information.

**Why Macro F1, not accuracy:** a naive classifier that always predicts Neutral achieves ≈52% accuracy but zero recall on four of the five classes. Macro F1 averages the F1 score across all 5 classes with equal weight, making one correctly-identified Strong Drop economically equivalent to one correctly-identified Neutral.

---

## 5. Notebook 01 — EDA & Target Creation

### Key EDA Findings

| Finding | Value |
|---|---|
| Total samples (post-preprocessing) | ≈3,700 |
| Date range | 2010-04-01 → 2024-10-18 |
| Number of features | ≈25 |
| Neutral class share | ≈51.7% |
| Imbalance ratio (Neutral : Strong Drop) | ≈10:1 |
| Dummy classifier (most_frequent) Macro F1 | 0.109 |

### Why the Imbalance is Structural

Under a symmetric return distribution, the probability of falling in the Neutral band (within 0.5σ of the mean) is inherently higher than for the tail classes. With real fat-tailed return distributions, the Neutral band captures even more mass. **The imbalance is not a data collection artifact** — it reflects the true return distribution. Removing it or ignoring it would misrepresent the problem.

---

## 6. Notebook 02 — Imputation Methods

### The Imputation Leakage Problem

All imputers are **fitted on the training set only** and then applied to transform both training and test sets. Fitting the imputer on combined train+test would leak future regime statistics into training values — a violation of temporal causality.

```python
# Correct implementation
mice = IterativeImputer(max_iter=10, random_state=42)
X_train_imp = pd.DataFrame(mice.fit_transform(X_tr_missing), columns=feature_cols)  # fit + transform
X_test_imp  = pd.DataFrame(mice.transform(X_te_missing),     columns=feature_cols)  # transform only
```

### The Four Imputation Methods

#### Method 1 — Mean Imputation (univariate)
Replaces each NaN with the training-set column mean $\hat{\mu}_j$.

**Limitation:** introduces variance compression (every imputed entry at the same constant). On the MACD feature, mean imputation replaces time-varying momentum readings with a static neutral signal. The classifier cannot use these rows to learn high-MACD or low-MACD regime distinctions.

#### Method 2 — Median Imputation (univariate)
Replaces each NaN with the training-set column median. The median has a 50% breakdown point (robust to outliers), but for approximately symmetric technical indicators the results are nearly identical to mean imputation.

#### Method 3 — KNN Imputation (k=5)
For each row with a missing feature, finds its 5 nearest neighbors (by Euclidean distance over observed features) and imputes with the average of the missing feature across those neighbors.

**Why KNN underperforms on RSI:** in the full 25-dimensional feature space, two rows with identical RSI can be far apart (different macro regime, different volatility state), and two rows with different RSI can appear nearby. The high-dimensional distance metric does not align well with individual indicator values — the "local manifold" assumption breaks down.

#### Method 4 — MICE (Multivariate Imputation by Chained Equations) ✓ Winner
**Algorithm:** for each column $j$ with missing values, fit a BayesianRidge regression using all other columns $X_{-j}$ as predictors on the observed rows, then predict the missing entries. Cycle through all missing columns repeatedly for `max_iter=10` iterations until convergence.

**Why MICE wins so decisively on MACD:** the three MACD features have a near-deterministic algebraic structure:
$$\text{MACD} = \text{EMA}_{12} - \text{EMA}_{26}$$
$$\text{MACD\_hist} = \text{MACD} - \text{MACD\_signal}$$

When MICE builds the regression for missing `nasdaq_macd`, it uses `nasdaq_macd_signal` and `nasdaq_macd_hist` as predictors and automatically discovers the coefficient ≈1 on both — re-deriving the deterministic formula from data. The result: MAE of 0.294 vs 4.798 for mean imputation (a **16× improvement**).

### Reconstruction Quality Results (MAE, lower is better)

| Column | Mean | Median | KNN (k=5) | **MICE** | MICE vs Mean |
|---|---|---|---|---|---|
| `nasdaq_rsi_14` | 9.416 | 9.490 | 11.413 | **7.474** | −21% |
| `nasdaq_macd` | 4.798 | 4.800 | 4.423 | **0.294** | **−94%** |
| `nasdaq_macd_signal` | 4.485 | 4.507 | 4.089 | **0.261** | **−94%** |
| `nasdaq_macd_hist` | 1.261 | 1.257 | 1.248 | **0.089** | **−93%** |
| `nasdaq_bb_width` | 0.044 | 0.050 | 0.040 | **0.019** | −57% |
| `nasdaq_lr_lag1` | 0.013 | 0.013 | 0.012 | **0.004** | −69% |
| `nasdaq_lr_lag2` | 0.008 | 0.008 | 0.009 | 0.010 | **+25% (worse)** |
| `nasdaq_vol_5d` | 0.006 | 0.006 | 0.006 | **0.005** | −17% |

**Note on `nasdaq_lr_lag2`:** MICE marginally loses here because lag-2 return has near-zero correlation with all other features. The regression adds noise rather than signal — the global mean is the better estimate when no predictor is informative.

### Downstream F1 Comparison (RandomForestClassifier)

| Imputer | Macro F1 |
|---|---|
| **MICE** | **Best** |
| KNN (k=5) | Second |
| Mean | Third |
| Median | ~Equal to Mean |

The margin between imputers is smaller than the reconstruction gap because MCAR at 5% affects only ~185 rows per column — enough to matter for minority classes (180–230 total samples) but not enough to dramatically shift aggregate Macro F1.

**Selected imputer for Notebook 03: MICE.**

---

## 7. Notebook 03 — Imbalanced Classification & Explainability

### Full Pipeline Architecture

```
df_missing
    → chronological split (80/20)
    → compute thresholds on train returns only
    → MICE fit on train_missing, transform both splits
    → StandardScaler fit on train_imputed, transform both splits
    → [Resampler — train only] → RandomForestClassifier(n_estimators=300)
```

All resampling is placed **inside** an `imblearn.Pipeline` to prevent synthetic samples from leaking into validation folds. Cross-validation uses `TimeSeriesSplit(n_splits=5)` — every validation fold is temporally posterior to its training fold.

### Evaluation Metrics

| Metric | Formula | Why it's used |
|---|---|---|
| **Macro F1** (primary) | $\frac{1}{K}\sum_k \frac{2P_kR_k}{P_k+R_k}$ | Equal weight to all 5 classes, penalising failure on minorities |
| Balanced Accuracy | Mean per-class recall | Simpler alternative to Macro F1; floors at 0.20 for 5-class random |
| Cohen's κ | $\frac{p_o - p_e}{1-p_e}$ | Corrects for chance agreement; 0 = random, 1 = perfect |

### Baseline Benchmarks

| Benchmark | Macro F1 | What it represents |
|---|---|---|
| Dummy (most_frequent) | 0.109 | Always predicts Neutral — absolute floor |
| Dummy (stratified) | 0.210 | Random draws from class distribution — meaningful floor |
| Baseline RF (no imbalance handling) | 0.231 | Real model, no correction — confirms features carry some signal |

**Any real model must beat 0.210** (stratified dummy) to demonstrate learning beyond chance.

### Imbalance Strategies

#### Strategy 1 — Class Weights (Macro F1 = 0.187 — worst non-dummy)
Inversely scales each sample's contribution to Gini impurity by class frequency. The extreme weights (4.11× for Strong Rise, 0.40× for Neutral) destabilize the mild intermediate classes — F1[Mild Drop] = 0.000.

**When it works:** mild imbalance ratios (2:1 to 5:1) with clean class boundaries. Not suitable for 5-class ordinal problems with 10:1 imbalance.

#### Strategy 2 — RandomOverSampler (Macro F1 = 0.250)
Randomly duplicates minority samples until all classes match the majority count. Simple duplication provides some benefit (identical copies still shift Gini weights) but cannot generalize beyond the convex hull of observed minority samples.

#### Strategy 3 — SMOTE (Macro F1 = 0.289 ✓ Winner)
Generates *synthetic* minority samples by interpolating between existing minority-class neighbors:

$$x_{\text{new}} = x_i + \lambda \cdot (x_{nn} - x_i), \quad \lambda \sim U[0,1]$$

New points lie on the line segment between $x_i$ and a same-class neighbor $x_{nn}$. This expands the learned minority-class manifold beyond the original observations without sampling from the majority-class region.

**Why SMOTE outperforms RandomOverSampler:** F1[Strong Drop] rises from 0.063 (Baseline) to 0.124 (RandomOverSampler) to 0.237 (SMOTE). Synthetic interpolation produces diverse feature vectors the classifier can recognize on unseen data.

#### Strategy 4 — ADASYN (Macro F1 = 0.280)
Extends SMOTE with adaptive density: generates more synthetic samples near minority examples that are surrounded by majority neighbors (hard boundary regions) and fewer near well-separated minority examples. Very similar result to SMOTE here because the class boundaries are diffuse across the entire feature space — no clear "easy vs hard" regions exist.

#### Strategy 5 — RandomUnderSampler (Macro F1 = 0.194)
Discards majority samples until all classes match the rarest class (n=180). Reduces the training set from ≈2,960 to 900 samples — a 70% reduction that destroys useful majority-class signal at this dataset scale.

#### Strategy 6 — SMOTEENN (Macro F1 = 0.231)
Combines SMOTE oversampling with Edited Nearest Neighbors (ENN) cleaning. ENN removes any sample whose majority of $k$ neighbors belong to a different class. In a 5-class problem with overlapping, continuously blending class distributions, ENN removes legitimate Neutral boundary samples because their nearest neighbors are Mild class samples. F1[Neutral] collapses to 0.274 (vs 0.546 for SMOTE), dragging Macro F1 down to the baseline level despite better minority recall.

### Complete Strategy Comparison

| Strategy | Macro F1 | F1[0 Drop] | F1[1 MildD] | F1[2 Neutral] | F1[3 MildR] | F1[4 Rise] | κ |
|---|---|---|---|---|---|---|---|
| **SMOTE** | **0.289** | 0.237 | 0.087 | 0.546 | 0.312 | 0.263 | 0.158 |
| ADASYN | 0.280 | 0.204 | 0.114 | 0.522 | 0.259 | 0.303 | 0.139 |
| RandomOverSampler | 0.250 | 0.124 | 0.048 | 0.551 | 0.254 | 0.274 | 0.106 |
| Baseline (no handling) | 0.231 | 0.063 | 0.070 | 0.559 | 0.212 | 0.252 | 0.089 |
| SMOTEENN | 0.231 | 0.175 | 0.179 | 0.274 | 0.228 | 0.298 | 0.064 |
| RandomUnderSampler | 0.194 | 0.130 | 0.129 | 0.246 | 0.191 | 0.272 | 0.042 |
| Class Weights | 0.187 | 0.085 | 0.000 | 0.562 | 0.144 | 0.143 | 0.055 |
| GridSearchCV Tuned | 0.224 | — | — | — | — | — | 0.058 |

**Hardest class across all strategies: Mild Drop (class 1).** Its return range sits between Strong Drop and Neutral with no clean boundary — a structural property of the return distribution that imputation quality cannot resolve.

### SHAP Explainability

**Method:** `shap.TreeExplainer` — computes exact Shapley values for tree-based classifiers in $O(TLD^2)$ time (T trees, L leaves, D max depth).

#### Global Feature Importance (ranked by mean |SHAP|)

1. **`nasdaq_lr_lag1`** — the most recent NASDAQ log return is the single most informative feature. Positive lag1 tilts toward Mild Rise; negative toward Mild Drop.
2. **`nasdaq_vol_5d`** — 5-day realized volatility increases the probability of **both** Strong Drop and Strong Rise simultaneously, pulling mass from Neutral. Elevated uncertainty increases the likelihood of extreme outcomes regardless of direction.
3. **`nasdaq_rsi_14`** — RSI ≤ 30 (oversold) tilts toward Strong Drop; RSI ≥ 70 (overbought) tilts toward Strong Rise; mid-range RSI supports Neutral.
4. **`nasdaq_macd_hist`** — momentum acceleration: positive and growing histogram supports Mild/Strong Rise; negative and declining supports Mild/Strong Drop. This is the feature where MICE imputation quality most directly affects downstream classification.
5. **Macro / FX variables** (`us_rates_%`, `CPI`, `GDP`, `eur_usd`, `usd_chf`) — consistent with Assignment 1: statistically weak but non-zero signal at daily frequency; appear in the lower half of SHAP importance.

#### Per-Class SHAP Highlights

**Strong Drop (class 0):**
- High `nasdaq_vol_5d` → increases Strong Drop probability
- High `nasdaq_lr_lag1` → *reduces* Strong Drop probability (momentum reversal)
- Low `nasdaq_rsi_14` (oversold) → increases Strong Drop probability

**Strong Rise (class 4):**
- High `nasdaq_vol_5d` → increases Strong Rise probability (same as Strong Drop — volatility is symmetric in its effect on extremes)
- High `nasdaq_lr_lag1` → *increases* Strong Rise probability (momentum continuation)

**Asymmetry finding:** Strong Drop and Strong Rise beeswarm panels are not mirror images. The divergence in feature rankings between the two quantifies the well-known asymmetry of financial markets — crashes are sharper, faster, and driven by different dynamics (fear, deleveraging) than rallies.

### Confusion Matrix Error Pattern

- **True Strong Drop → most often predicted as Neutral or Mild Drop.** Extreme events are systematically under-predicted.
- **True Strong Rise → most often predicted as Neutral or Mild Rise.** Same failure mode.
- **SMOTE vs Baseline:** Strong Drop diagonal rises from ≈0.04 to ≈0.16 while Neutral diagonal is preserved (≈0.54 vs ≈0.57). SMOTE improves minority recall without catastrophic majority collapse.

**Financial cost asymmetry:** missing a Strong Drop (predicting Neutral when a crash is incoming) is far more costly than a false alarm. The confusion matrix reveals that all strategies remain biased toward under-predicting extreme events — a conservative bias that limits utility for tail-risk management.

---

## 8. Key Results Summary

| Task | Method | Metric | Value |
|---|---|---|---|
| Reconstruction | MICE | MAE (MACD) | 0.294 (vs 4.798 for Mean) |
| Reconstruction | MICE | MAE (MACD hist) | 0.089 (vs 1.261 for Mean) |
| Classification | SMOTE + RF | Macro F1 | 0.289 |
| Classification | SMOTE + RF | Directional accuracy (Strong Drop) | F1 = 0.237 (vs 0.063 baseline) |
| Classification floor | Stratified dummy | Macro F1 | 0.210 |
| Classification ceiling | Best possible | κ (Cohen's) | 0.158 — modest real discriminative power |
| SHAP — top feature | `nasdaq_lr_lag1` | Global importance rank | #1 across all classes |
| SHAP — top variance feature | `nasdaq_vol_5d` | Symmetric effect on extremes | Increases both Strong Drop and Strong Rise |

---

## 9. Known Issues & Discrepancies

### 1. Documentation claims SMOTEENN wins — SMOTE is the actual winner

The Notebook 03 conclusions cell and original `Project_Memorie.md` both state SMOTEENN produces the best overall Macro F1. This is **factually incorrect** based on the actual output data.

From `outputs/reports/03_strategy_comparison.csv`:
- SMOTE Macro F1 = **0.2892** (winner)
- SMOTEENN Macro F1 = 0.2307 (barely above baseline)
- `performance_summary.csv` explicitly records "Best strategy: SMOTE"

**Root cause:** ENN's cleaning step removes legitimate Neutral boundary samples in a 5-class ordinal problem with heavily overlapping distributions, collapsing Neutral F1 from 0.546 (SMOTE) to 0.274 (SMOTEENN) and offsetting the improved minority recall.

### 2. GridSearchCV applied to SMOTEENN, not the actual winner

Hyperparameter tuning was applied to SMOTEENN (Macro F1 = 0.231 tuned) rather than SMOTE (Macro F1 = 0.289 untuned). This is a methodological inconsistency — the best strategy was not optimized.

Additionally, CV fold sizes (~2,220 training samples per fold) produce noisier estimates than the final holdout, and the best hyperparameters found on those smaller folds may not transfer. `class_weight` was correctly excluded from the grid to avoid double-penalizing the majority.

**Recommendation for future work:** run `GridSearchCV` on the SMOTE pipeline.

### 3. Class Weights underperforms the stratified dummy

Class Weights Macro F1 = 0.187 < Dummy (stratified) 0.210. The extreme weight ratios (4.11× for Strong Rise vs 0.40× for Neutral) destabilize intermediate classes — Mild Drop is never predicted (F1 = 0.000). This is the worst non-trivial strategy in the experiment.

---

## 10. How to Run

```bash
cd Assignment2

# Notebook 01: EDA and target creation
jupyter notebook notebooks/01_EDA_and_Target_Creation.ipynb

# Notebook 02: imputation methods benchmark
jupyter notebook notebooks/02_Imputation_Methods.ipynb

# Notebook 03: imbalanced classification and SHAP
jupyter notebook notebooks/03_Imbalanced_Classification_&_Explainability.ipynb
```

Notebooks must be run in order (01 → 02 → 03). The `data/df_clean.parquet` and `data/df_missing.parquet` files are pre-generated and included — you can run Notebooks 02 and 03 directly if Notebook 01 has already been executed and the parquet files are present.

---

## 11. Dependencies

| Package | Version | Purpose |
|---|---|---|
| `scikit-learn` | 1.6.0 | Random Forest, imputers, scalers, TimeSeriesSplit, GridSearchCV |
| `imbalanced-learn` | — | SMOTE, ADASYN, RandomOverSampler, RandomUnderSampler, SMOTEENN, imblearn.Pipeline |
| `shap` | — | TreeExplainer for SHAP values |
| `pandas` | 2.2.3 | Data loading and manipulation |
| `numpy` | 2.4.4 | Numerical arrays |
| `matplotlib` / `seaborn` | latest | Figures and heatmaps |
| `joblib` | 1.5.3 | Model persistence (`best_pipeline.joblib`) |
| `pyarrow` / `fastparquet` | — | Parquet file I/O |

**Note:** `sklearn.impute.IterativeImputer` (MICE) is still experimental in scikit-learn and requires an explicit import:

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
```

---

*Course: Advanced Machine Learning — Assignment 2: Data Imputation, Imbalanced Classification & Explainability*
