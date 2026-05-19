# Assignment 2 — Imbalanced Data, Imputation Methods, and Multi-class Classification

## Project Overview

Assignment 2 reuses the same financial dataset from Assignment 1 (NASDAQ-100 and related assets) but converts the regression target (continuous log return) into a **5-class ordinal classification problem**. The three new challenges introduced are:

1. **Missing data**: controlled MCAR injection into technical indicator features and a comparison of four imputation methods
2. **Class imbalance**: the Neutral class dominates ≈52% of the data; extreme classes (Strong Drop, Strong Rise) account for only ≈5–6% each
3. **Explainability**: SHAP TreeExplainer to identify which features drive each class prediction

The project is structured as three sequential notebooks backed by a modular `utils/` library.

```text
.
├── notebooks/
│   ├── 01_EDA_and_Target_Creation.ipynb
│   ├── 02_Imputation_Methods.ipynb
│   └── 03_Imbalanced_Classification_&_Explainability.ipynb
├── utils/
│   ├── data_loader.py       — preprocessing, feature engineering, target creation, MCAR injection
│   ├── imputation.py        — imputer catalogue, reconstruction quality, downstream F1 comparison
│   ├── imbalance.py         — resampling strategy catalogue, evaluation
│   └── evaluation.py        — plotting utilities (bar charts, confusion matrices, heatmaps)
├── data/
│   ├── financial_regression.csv    — raw input
│   ├── df_clean.parquet            — preprocessed without artificial missingness (ground truth)
│   └── df_missing.parquet          — preprocessed with 5% MCAR injected into 8 columns
└── outputs/
    ├── figures/                    — all saved plots
    ├── models/best_pipeline.joblib — persisted best tuned pipeline
    └── reports/                    — CSV result tables
```

---

## Problem Framing

### From regression to classification

Assignment 1 predicted the **continuous** NASDAQ log return $r_t$. Assignment 2 converts the next-day log return $r_{t+1}$ into a **5-class ordinal target** using mean-and-sigma thresholds:

$$t_1 = \mu - 1.5\sigma, \quad t_2 = \mu - 0.5\sigma, \quad t_3 = \mu + 0.5\sigma, \quad t_4 = \mu + 1.5\sigma$$

| Class | Label | Return range | Approx. share |
| --- | --- | --- | --- |
| 0 | Strong Drop | $r < t_1$ | ≈6.3% |
| 1 | Mild Drop | $t_1 \le r < t_2$ | ≈16.5% |
| 2 | Neutral | $t_2 \le r \le t_3$ | ≈51.7% |
| 3 | Mild Rise | $t_3 < r \le t_4$ | ≈20.7% |
| 4 | Strong Rise | $r > t_4$ | ≈4.9% |

The precise imbalance ratios (Neutral : each class) are: **8.1:1 vs Strong Drop**, **3.1:1 vs Mild Drop**, **2.5:1 vs Mild Rise**, **10.4:1 vs Strong Rise**. A naive classifier that always predicts Neutral achieves ≈52% accuracy but **zero recall on the economically important extreme classes**. This is why Macro F1 (not accuracy) is the primary metric — it weights each class equally regardless of frequency.

**Leakage prevention on thresholds**: in Notebook 01 thresholds are computed on the full series for descriptive visualization only. In Notebooks 02 and 03 thresholds are always recomputed on the **training split returns only** and applied verbatim to the test split — the labels never depend on future return information.

---

## Data & Feature Engineering (`utils/data_loader.py`)

### Preprocessing pipeline (identical to Assignment 1)

1. Load raw CSV, normalize column names, sort by date
2. Forward-fill macro variables (GDP quarterly, CPI monthly, US rates) — last known value interpretation
3. Drop market holidays (rows where NASDAQ and SP500 close are both NaN)
4. Compute log returns: $r_t = \log(P_t / P_{t-1})$ for all 7 close-price columns

### New technical indicators (NASDAQ only)

| Feature | Formula / Window | Meaning |
| --- | --- | --- |
| `nasdaq_rsi_14` | RSI(14), Wilder EMA ($\alpha = 1/14$) | Momentum oscillator 0–100 |
| `nasdaq_macd` | EMA(12) − EMA(26) | Trend/momentum line |
| `nasdaq_macd_signal` | EMA(9) of MACD | Signal smoothing |
| `nasdaq_macd_hist` | MACD − signal | Momentum acceleration |
| `nasdaq_bb_upper/lower` | 20-day rolling mean ± 2 std | Bollinger band boundaries |
| `nasdaq_bb_width` | (upper − lower) / mid | Relative band width = local volatility proxy |
| `nasdaq_lr_lag1/2/3/5` | `nasdaq_log_return.shift(k)` | Lagged own returns |
| `nasdaq_vol_5d` / `_20d` | rolling(5/20).std() on log return | Short/medium realized volatility |
| `sp500_lr_lag1`, `gold_lr_lag1` | `{asset}_log_return.shift(1)` | Cross-asset lag-1 returns |

These are computed by `compute_technical_indicators()` and represent the feature space for classification.

### MCAR missingness injection

5% of values are set to `NaN` independently at random per column (MCAR = **Missing Completely At Random**) in 8 technical indicator columns:

```text
nasdaq_rsi_14, nasdaq_macd, nasdaq_macd_signal, nasdaq_macd_hist,
nasdaq_bb_width, nasdaq_lr_lag1, nasdaq_lr_lag2, nasdaq_vol_5d
```

MCAR is the strongest form of ignorable missingness — the probability that a value is missing does not depend on the value itself or on any other variable. This is the correct experimental design for an imputation benchmark because it gives a known ground truth for reconstruction evaluation.

The original `df_clean` is saved separately so reconstruction MAE/RMSE can be computed against the true values.

### Chronological split

```python
train = first 80% of rows by date
test  = last  20% of rows by date
```

Financial features include rolling windows, lagged values, and autocorrelated targets — a random stratified split would leak temporal context across the boundary. **No shuffling is ever applied** in any split or cross-validation in this project.

---

## Notebook 01 — EDA & Target Creation

### Key EDA findings

| Finding | Value |
| --- | --- |
| Total samples (post-preprocessing) | ≈3,700 |
| Date range | 2010-04-01 → 2024-10-18 |
| Number of features | ≈25 |
| Neutral class share | ≈51.7% |
| Imbalance ratio (Neutral : Strong Drop) | ≈10:1 |
| Dummy classifier (most_frequent) Macro F1 | 0.109 |

### Why the imbalance is structural

The 5-class partition is defined by 1-sigma bands around the mean return. Under a symmetric return distribution:

- P(|r - μ| < 0.5σ) ≈ 38% (Neutral band)
- P(0.5σ < |r - μ| < 1.5σ) ≈ 38% (Mild bands)
- P(|r - μ| > 1.5σ) ≈ 13% (Strong bands)

With real fat-tailed return distributions the Neutral band captures even more mass. The imbalance is **not a data collection artifact** — it reflects the true return distribution. Any classifier must be designed to handle it explicitly.

---

## Notebook 02 — Imputation Methods

### The missing data problem and the MCAR assumption

Before choosing an imputation method we need to understand *why* values are missing, because the missingness mechanism determines which imputers are valid.

There are three types of missingness:

| Mechanism | Definition | Example | Imputable? |
| --- | --- | --- | --- |
| **MCAR** (Missing Completely At Random) | P(missing) is the same for all rows, independent of any variable | Random 5% erasure | Yes — unbiased |
| **MAR** (Missing At Random) | P(missing) depends on *observed* variables but not on the missing value itself | RSI missing more often on high-volatility days (we can observe volatility) | Yes — with multivariate methods |
| **MNAR** (Missing Not At Random) | P(missing) depends on the value that is missing | Extreme MACD readings failing to report | Biased — no clean fix |

In this project we **artificially inject MCAR missingness**: 5% of values per column are erased uniformly at random, independent of any other variable or the missing value itself. This is the experimentally cleanest scenario — it guarantees unbiased imputation is possible in principle and gives us a known ground truth to measure reconstruction quality against.

**Why MCAR and not MNAR?** Real financial data often has MNAR-style gaps (feeds go down during volatile periods, thinly traded instruments have stale prices). MCAR is chosen here not to model reality but to enable a fair, controlled benchmark: we know exactly which values we removed and can measure recovery accuracy precisely. If the missingness were truly informative (MNAR), imputation alone could not recover the signal — you would need to model the missingness mechanism itself.

---

### The imputation leakage problem

All imputers are **fitted on the training set only** and then applied to transform both the training set and the test set. This is not a technicality — it is a fundamental requirement to avoid data leakage.

**What would go wrong if you fit on train+test combined?**

Fitting the mean imputer on combined data gives: $\hat{\mu}_j = \frac{1}{n_{train}+n_{test}} \sum_{all} x_{ij}$. This mean incorporates test-set statistics into the imputed training values. A MICE regressor fit on both sets learns relationships from test-set rows. Both cases violate temporal causality: at training time, the model has "seen" future data.

For time series in particular, leakage via imputer fitting can be subtle. The test period (late 2022–2024) has a different volatility regime than the training period. Fitting the mean across both would use the average of two different regimes. Instead we fit on the training regime only and accept that imputed test values are based on a slightly different distribution — this is the correct, realistic setup.

```python
mice = IterativeImputer(max_iter=10, random_state=42, tol=1e-3)
X_train_imp = pd.DataFrame(mice.fit_transform(X_tr_missing), columns=feature_cols)  # fit + transform train
X_test_imp  = pd.DataFrame(mice.transform(X_te_missing),     columns=feature_cols)  # transform only
```

---

### Method 1 — Mean imputation (univariate)

**Algorithm**: replace each NaN in column $j$ with the training-set column mean $\hat{\mu}_j$:

$$\hat{x}_{ij} = \hat{\mu}_j = \frac{1}{n_{obs,j}} \sum_{i: x_{ij} \text{ observed}} x_{ij}$$

**Statistical justification**: the sample mean is the minimum-MSE estimator of a missing value when the feature is i.i.d. and no other information is available. It is the correct choice under a Gaussian marginal and under complete ignorance of all other features.

**What it gets wrong — variance compression**: replacing 5% of values with the constant $\hat{\mu}_j$ reduces the empirical variance of the feature. If $X \sim N(\mu, \sigma^2)$ and fraction $p$ of values are replaced by $\mu$, the observed variance becomes approximately $(1-p)\sigma^2$. At $p=0.05$ this is a 5% compression — modest but systematic. More importantly, the *information content* at those 5% positions is zero: the imputed value is the same constant regardless of what the market was doing on that day. The classifier cannot use those rows to learn high-MACD or low-MACD regime distinctions because every imputed entry looks identical.

**Geometric view**: in the scatter plot of true vs imputed MACD values, mean imputation produces a horizontal line — all imputed values sit at $y = \hat{\mu}$. The variance along the x-axis (true values) is real, but the imputer contributes no matching variance along the y-axis. The more the feature varies temporally, the worse mean imputation is.

---

### Method 2 — Median imputation (univariate)

**Algorithm**: replace each NaN with the training-set column median $\hat{m}_j$.

**Difference from mean**: the median has a **breakdown point of 50%** — it can withstand up to 50% of observations being corrupted outliers before it becomes arbitrarily wrong (the mean's breakdown point is 0%). For RSI (bounded 0–100, approximately symmetric), the median and mean are nearly identical and the results confirm this: median MAE ≈ mean MAE on every column.

**When median would meaningfully outperform mean**: features with heavy-tailed or skewed distributions (e.g., volume-based indicators, commodity prices) where a small number of extreme values pull the mean far from the typical value. In this project the feature distributions are close enough to symmetric that the two univariate methods perform almost identically.

Both Mean and Median share the same fundamental flaw: they are **univariate** — they use no information from other features in the same row. A row where `nasdaq_macd_signal = +6.2` (strongly trending up) gets the same imputed MACD value as a row where `nasdaq_macd_signal = −5.1` (strongly trending down). This is precisely the information KNN and MICE try to exploit.

---

### Method 3 — KNN imputation (k=5)

**Algorithm**:

1. For each test row $x$ with one or more missing features, compute the Euclidean distance to every training row using only the *non-missing* feature dimensions.
2. Find the $k=5$ closest training rows ("neighbors").
3. For each missing feature, impute with the average of that feature across the 5 neighbors.

$$\hat{x}_{ij} = \frac{1}{k} \sum_{l \in \text{kNN}(i)} x_{lj}$$

**Core assumption**: rows that are nearby in the observed feature space are also similar in the missing feature. This is a **local manifold assumption** — the data lies on a lower-dimensional manifold where proximity in any observed projection generalizes to unobserved dimensions.

**Why KNN fails on RSI in this dataset**: RSI is a momentum oscillator derived from the ratio of average gains to average losses over a 14-day window. The other 25 features include cross-asset returns, macro variables, and Bollinger Band measures. In the full 25-dimensional space used for distance computation, two rows with identical RSI can be far apart (different macro regime, different vol state), and two rows with very different RSI can be nearby (same macro and vol regime, but different recent price path). The 25 non-RSI features dilute the distance metric for RSI purposes — the neighbors found are not RSI-similar, so the KNN average is noisy. The global mean is a better estimate precisely because it ignores the irrelevant proximity structure.

**Computational cost**: $O(n_{test} \times n_{train} \times p)$ at prediction time where $p$ is the number of features. With $n_{train} \approx 2960$ and $p = 26$, this is manageable but grows quadratically with dataset size — KNN does not scale to millions of rows.

**Where KNN works well**: datasets with strong local cluster structure (e.g., users with similar purchase histories all share similar missing rating values). In financial data, local temporal clusters do exist (momentum regimes), but the 26-dimensional feature space captures many dimensions of variation that are orthogonal to any individual technical indicator.

---

### Method 4 — MICE (Multivariate Imputation by Chained Equations)

This is the theoretically strongest method and the one selected for Notebook 03.

#### The algorithm step by step

**Initialization**: fill all missing values with their column means. This gives a complete matrix to start with.

**Iteration** (repeated `max_iter=10` times): for each column $j$ that has missing values:

1. Partition the rows into: observed rows (where $x_j$ was not missing) and missing rows (where $x_j$ was NaN).
2. Use all other columns as features: $X_{-j}$ = the data matrix with column $j$ removed.
3. Fit a **BayesianRidge regression** on the observed rows: $x_j^{obs} = X_{-j}^{obs} \beta + \epsilon$.
4. Predict the missing values: $\hat{x}_j^{miss} = X_{-j}^{miss} \hat{\beta}$.
5. Replace the missing entries of column $j$ with these predictions.
6. Move to the next column that has missing values and repeat with the updated matrix.

After cycling through all missing columns once, that is one full iteration. Repeat for `max_iter` iterations.

**Why "Chained Equations"**: each column is imputed by its own regression equation, and these equations are chained — the regression for column $k$ uses the already-updated imputed values of columns $1, ..., k-1$ from earlier in the current cycle. The equations depend on each other through the shared imputed values, forming a Markov chain over the space of complete-data matrices.

**Convergence**: under regularity conditions (linear relationships, MCAR/MAR missingness), the sequence of imputed matrices converges to the true conditional expectations $E[x_j | X_{-j}]$. In practice with `max_iter=10` the algorithm converges well for linear feature relationships.

**Computational cost**: $O(n \times p^2 \times \text{max\_iter})$ — the $p^2$ factor comes from fitting $p$ regressions each using $p-1$ features. With $p=26$ and $n \approx 2960$, this is fast. The cost grows as $p^2$ with the number of features.

**sklearn implementation note**: `IterativeImputer` is still marked experimental in sklearn and requires an explicit import flag:

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
```

#### Why MICE wins so decisively on the MACD family

The MACD features have a near-deterministic structure that MICE can exploit directly:

$$\text{MACD} = \text{EMA}_{12} - \text{EMA}_{26}$$
$$\text{MACD\_signal} = \text{EMA}_9(\text{MACD})$$
$$\text{MACD\_hist} = \text{MACD} - \text{MACD\_signal}$$

These three quantities are algebraically related: $\text{MACD\_hist} = \text{MACD} - \text{MACD\_signal}$, so $\text{MACD} = \text{MACD\_signal} + \text{MACD\_hist}$.

When MICE builds the regression for missing `nasdaq_macd` values, it uses `nasdaq_macd_signal` and `nasdaq_macd_hist` as predictors. The regression automatically discovers:

$$\hat{\text{MACD}} \approx 1.0 \cdot \text{MACD\_signal} + 1.0 \cdot \text{MACD\_hist} + \epsilon$$

The fitted coefficient is ≈1 on both predictors and the residual is near zero — the regression *re-derives the deterministic formula from data*. This is why MICE achieves MAE 0.294 (essentially recovering the true value) while Mean achieves MAE 4.798 (replacing every entry with the same constant regardless of regime).

The same logic applies in reverse: when MICE imputes a missing `nasdaq_macd_signal`, it uses `nasdaq_macd` and `nasdaq_macd_hist` as predictors and recovers $\text{signal} \approx \text{MACD} - \text{hist}$ automatically.

**This is the ideal case for MICE**: features that are deterministic (or near-deterministic) functions of each other. Real financial datasets often have this structure in their technical indicators because many indicators are derived from the same underlying price series.

#### Why MICE underperforms on `nasdaq_lr_lag2`

`nasdaq_lr_lag2` is $r_{t-2}$ — the log return two days ago. Its relationship to all other features is:

- $r_{t-1}$ (lag1) carries some autocorrelation with lag2, but NASDAQ autocorrelations are tiny (≈0.01–0.05)
- All other features (RSI, MACD, Bollinger, cross-asset returns) have even weaker correlations with lag2 at the daily frequency
- The MICE regression for lag2 uses all other 25 features but gets a very low $R^2$ — essentially fitting noise

With low predictive signal, the MICE regression collapses toward its prior (the mean), producing imputed values close to the mean but with added noise from the near-zero coefficient estimates. The result is slightly worse than just using the mean directly. **This illustrates a general principle**: MICE only helps when the missing feature is predictable from the observed features. If the missing column is nearly uncorrelated with everything else, MICE adds noise rather than signal.

---

### Evaluation protocol

**What we measure and why**:

- **Ground truth**: `df_clean` (the pre-injection dataset) is saved before any NaN are introduced. At evaluation time, we compare imputed values against `df_clean` only at the positions that were artificially set to NaN.
- **Metric — MAE**: $\frac{1}{n_{miss}} \sum_{(i,j) \in \text{missing}} |\hat{x}_{ij} - x_{ij}^{true}|$. MAE is used over RMSE because it is robust to occasional large imputation errors (outlier true values that no method can predict well). RMSE would square these errors and inflate the comparison.
- **Test-set only**: imputation quality is measured on the test split (imputer fit on train) to correctly measure generalization. Measuring on the training set would understate the error for MICE, which can memorize training-set relationships.

**The two-stage evaluation**:

1. **Reconstruction quality** (columns of the MAE table): does the imputer accurately recover the numerical value of the missing entry?
2. **Downstream F1** (the bar chart): does better reconstruction translate into better classification?

These are related but distinct questions. It is possible (in principle) for a more accurate imputer to produce worse downstream F1 if the reconstruction introduces different systematic errors. In practice here, reconstruction quality and downstream F1 are correlated — MICE wins both.

---

### Reconstruction quality results

**MAE per corrupted column (lower is better):**

| Column | Mean | Median | KNN (k=5) | **MICE** | MICE improvement over Mean |
| --- | --- | --- | --- | --- | --- |
| `nasdaq_rsi_14` | 9.416 | 9.490 | 11.413 | **7.474** | −21% |
| `nasdaq_macd` | 4.798 | 4.800 | 4.423 | **0.294** | **−94% (16×)** |
| `nasdaq_macd_signal` | 4.485 | 4.507 | 4.089 | **0.261** | **−94% (17×)** |
| `nasdaq_macd_hist` | 1.261 | 1.257 | 1.248 | **0.089** | **−93% (14×)** |
| `nasdaq_bb_width` | 0.044 | 0.050 | 0.040 | **0.019** | −57% |
| `nasdaq_lr_lag1` | 0.013 | 0.013 | 0.012 | **0.004** | −69% |
| `nasdaq_lr_lag2` | 0.008 | 0.008 | 0.009 | 0.010 | +25% (worse) |
| `nasdaq_vol_5d` | 0.006 | 0.006 | 0.006 | **0.005** | −17% |

**Summary of what the table reveals**:

- **MACD family** (rows 2–4): MICE wins by 14–17×. The deterministic algebraic structure is fully exploited.
- **`nasdaq_bb_width`** and **`nasdaq_lr_lag1`**: MICE wins by 57–69%. These have moderate correlations with other features (Bollinger width correlates with volatility; lag1 has weak autocorrelation with other lags).
- **`nasdaq_rsi_14`**: MICE wins by only 21%; KNN is worst. RSI has moderate correlation with MACD but the regression signal is weaker than for the MACD family because RSI involves a non-linear transformation (ratio of average gains to losses).
- **`nasdaq_lr_lag2`**: MICE loses marginally (+25% worse than Mean). Near-zero correlation with everything else means the MICE regression adds noise.
- **KNN never wins and underperforms Mean on RSI**: the high-dimensional distance metric does not align well with individual indicator values.

**Practical implication (from scatter plot analysis)**: mean imputation of MACD places all imputed values at a constant $\approx 0$ — every imputed entry looks like a "neutral trend" day regardless of actual market conditions. MICE recovers the actual MACD value for the missing entries, preserving the time-varying momentum information. This distinction matters because MACD is one of the top-5 globally important SHAP features: destroying its signal at 5% of training positions measurably degrades classification performance.

---

### Downstream F1 comparison

A `RandomForestClassifier(n_estimators=200)` trained on imputed training data and evaluated on imputed test data:

| Imputer | Macro F1 | Observation |
| --- | --- | --- |
| **MICE** | **best** | Preserves time-varying momentum signal; all features informative |
| KNN (k=5) | second | Partial local-structure recovery; close to MICE on most columns |
| Mean | third | Constant imputation destroys feature variance at missing positions |
| Median | fourth (≈Mean) | Nearly identical to Mean for approximately symmetric indicators |

**Why the margin between imputers is smaller than the reconstruction gap**: at 5% MCAR, only 5% of each column's values are affected. The 95% of correctly observed values still carry full signal. The classifier learns primarily from complete rows; imputation quality mainly affects the 5% of rows with at least one missing feature. For a dataset of ≈3,700 rows, this is roughly 185 rows per imputed column — enough to matter for minority classes (which have only 180–230 total samples) but not enough to dramatically shift aggregate Macro F1.

**Why Mild Drop (class 1) is the hardest for all imputers**: class 1 sits between Neutral (class 2) and Strong Drop (class 0) in the return distribution. Its boundaries overlap with both neighbors in almost every feature dimension. Imputation quality has no effect on this overlap — it is a property of the underlying return distribution and cannot be resolved by filling in missing values more accurately. This is confirmed by the per-class F1 heatmap: class 1 has the lowest F1 across all imputers, and the spread between best and worst imputer is smaller for class 1 than for the extreme classes.

**Selected for Notebook 03: MICE / IterativeImputer.** The theoretical justification (recovery of near-deterministic feature relationships), the empirical reconstruction results (14–17× improvement on MACD), and the downstream F1 advantage all point to the same choice. If computational efficiency were the primary constraint at scale (millions of rows), KNN would be the reasonable alternative; the reconstruction quality gap on non-MACD columns is small.

---

## Notebook 03 — Imbalanced Classification & Explainability

### Full pipeline architecture

```text
df_missing
    → chronological split (80/20)
    → compute thresholds on train returns only
    → MICE fit on train_missing, transform both splits
    → StandardScaler fit on train_imputed, transform both splits
    → [Resampler — train only] → RandomForestClassifier(n_estimators=300)
```

All resampling lives **inside** an `imblearn.Pipeline` so synthetic samples are never generated from test data. Cross-validation uses `TimeSeriesSplit(n_splits=5)` — each fold's validation set is always temporally posterior to its training set.

---

### Step 1 — Evaluation metrics: what we measure and why

Before explaining the models, it is critical to understand the three metrics used and why each one is needed.

**Accuracy** (not used as primary metric): the fraction of correct predictions. With 51.7% Neutral samples, a model that always predicts Neutral scores 51.7% accuracy. This looks reasonable but is completely useless — it has zero recall on four of the five classes.

**Macro F1** (primary): computes F1 independently for each class and averages with equal weight:

$$\text{Macro F1} = \frac{1}{K} \sum_{k=0}^{K-1} \frac{2 \cdot P_k \cdot R_k}{P_k + R_k}$$

where $P_k$ = precision for class k (fraction of class-k predictions that are correct) and $R_k$ = recall for class k (fraction of true class-k samples that are correctly predicted). Equal weight means that getting one more Strong Drop correct matters as much as getting one more Neutral correct, even though Neutral appears 10× more often. This makes Macro F1 the right metric when all classes matter equally — which is the economic argument here (a missed crash matters as much as a missed rally).

**Balanced Accuracy**: the mean of per-class recall = $\frac{1}{K}\sum_k R_k$. Equivalent to Macro Recall. A model that achieves 20% recall in every class gets Balanced Accuracy = 0.20. The Dummy (most_frequent) classifier has recall 1.0 for Neutral and 0.0 for all others → Balanced Accuracy = 1/5 = 0.200.

**Cohen's κ**: corrects accuracy for the agreement expected by chance alone:

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

$p_o$ is the observed proportion of agreement (accuracy); $p_e$ is the expected proportion if both the true labels and predictions were independent draws from the marginal class distribution. κ = 0 means the model does no better than a random guesser that knows the class frequencies; κ = 1 means perfect prediction. κ < 0 means the model is actively worse than random. The Baseline RF's κ = 0.089 means it barely improves on chance — confirming that without imbalance handling it has learned almost nothing useful about minority classes.

---

### Step 2 — Dummy classifiers: the floor benchmarks

A **Dummy classifier** is a model that ignores the features entirely. It makes predictions based solely on the label distribution seen during training. Dummies do not learn anything — they are used purely to establish the floor that any real model must beat.

**Why do we need a floor?** Without a reference point it is impossible to know whether a Macro F1 of 0.231 is "good" or "bad". If a coin flip achieves 0.210, then 0.231 is barely above random. If the coin flip achieves 0.050, then 0.231 represents substantial learning. The dummy defines the floor.

#### Dummy (most_frequent) — Macro F1 = 0.109

This dummy always predicts the most common class, which is Neutral (class 2).

- Precision for Neutral: n_correct_neutral / n_predicted_neutral = n_neutral / n_total ≈ 0.517 (it predicts Neutral for every sample, and 51.7% of all samples are truly Neutral)
- Recall for Neutral: n_correct_neutral / n_true_neutral = n_neutral / n_neutral = 1.0 (it never misses a Neutral because it always predicts Neutral)
- F1 for Neutral = 2 × 0.517 × 1.0 / (0.517 + 1.0) ≈ 0.682
- F1 for all other classes = 0.000 (zero recall — no sample of class 0, 1, 3, 4 is ever predicted correctly)
- Macro F1 = (0.682 + 0 + 0 + 0 + 0) / 5 ≈ **0.136** (actual value ≈ 0.109 due to test-set class proportions differing slightly from training)

This is the absolute floor. It tells you: if your model cannot beat 0.109, it is literally doing worse than "always say Neutral."

#### Dummy (stratified) — Macro F1 = 0.210

This dummy randomly samples a prediction from the training class distribution. If the training set is 51.7% Neutral, then 51.7% of this dummy's predictions are Neutral. The predictions are drawn independently of any feature.

For a class with true proportion $p_k$:

- Precision ≈ $p_k$ (fraction of predicted-class-k that truly are class k ≈ the marginal probability)
- Recall ≈ $p_k$ (fraction of true-class-k samples predicted as class k ≈ the marginal probability, since predictions are random)
- F1 for class k ≈ $p_k$

So Macro F1 ≈ mean($p_k$) across all 5 classes ≈ mean(0.063, 0.165, 0.517, 0.207, 0.049) ≈ **0.200** (actual 0.210 due to exact test proportions).

The stratified dummy is the more stringent and meaningful floor. **Any supervised model that cannot beat 0.210 has failed to learn anything useful** — it is being outperformed by a random coin flip that knows only the class frequencies.

---

### Step 3 — The Baseline Random Forest

The **Baseline** is not a dummy. It is a real, trained Random Forest classifier (`n_estimators=300`) that sees all 26 features and genuinely learns from them — but without any correction for class imbalance.

#### What is a Random Forest?

A Random Forest is an **ensemble of decision trees**. Each tree:

1. Receives a **bootstrap sample** of the training data (random sampling with replacement, same size as original)
2. At each split node, considers a random subset of features (typically $\sqrt{p}$ features)
3. Chooses the split that best separates the classes according to **Gini impurity**:

$$\text{Gini}(S) = 1 - \sum_{k=0}^{K-1} p_k^2$$

where $p_k$ is the fraction of class-k samples in the node. Gini = 0 means a pure node (all one class); Gini = 0.5 (for binary) or higher means maximum mixing.

The final prediction for a new sample is the majority vote across all 300 trees.

#### Why the Baseline RF still beats the dummy

It genuinely learns the feature space. It discovers, for example, that high `nasdaq_vol_5d` + negative `nasdaq_lr_lag1` is more often associated with negative returns (classes 0–1) than with Neutral. These partial relationships are real even if weak. The Baseline Macro F1 = 0.231 > Dummy stratified 0.210 confirms that 26 features do contain predictive signal about return direction.

#### Why the Baseline RF fails on minority classes

The Gini criterion is driven by the class distribution at each node. With 51.7% Neutral samples, any split that correctly separates more Neutral samples reduces Gini more than a split that correctly separates Strong Drop samples. Concretely:

- A split that moves 100 Neutral samples to a pure node reduces Gini significantly.
- A split that moves 10 Strong Drop samples to a pure node reduces Gini only slightly.

The RF's 300 trees each independently optimize Gini, so the aggregate decision boundary is heavily biased toward correctly predicting Neutral. The result in the numbers: F1[Neutral] = 0.559 but F1[Strong Drop] = 0.063. The model learned to be a very good Neutral predictor and a poor everything-else predictor. This is not a flaw in the model — it is the mathematically correct response to the training data as presented.

---

### Step 4 — Imbalance strategies from first principles

Each strategy below is a different answer to the same question: how do we change the training data or the loss function so that minority classes are treated fairly?

---

#### Strategy 1 — Class Weights (result: 0.187 — worst performer)

**What it does**: instead of modifying the data, it modifies the loss function. Each sample's contribution to the Gini impurity is multiplied by a weight inversely proportional to its class frequency:

$$w_k = \frac{n_{total}}{K \cdot n_k}$$

With our class distribution:

| Class | n_k | Weight $w_k$ |
| --- | --- | --- |
| Strong Drop (0) | 231 | 3700 / (5 × 231) ≈ **3.20** |
| Mild Drop (1) | 594 | 3700 / (5 × 594) ≈ **1.25** |
| Neutral (2) | 1871 | 3700 / (5 × 1871) ≈ **0.40** |
| Mild Rise (3) | 748 | 3700 / (5 × 748) ≈ **0.99** |
| Strong Rise (4) | 180 | 3700 / (5 × 180) ≈ **4.11** |

A misclassified Strong Rise sample now costs 4.11× what a misclassified Neutral sample costs. The RF's Gini calculation sees these weights, so splits that correctly identify Strong Rise or Strong Drop become much more attractive.

**Why it fails here**: the weight ratios are extreme (4.11× vs 0.40×). The RF aggressively adjusts its decision boundaries to capture the extreme classes, but in doing so it destabilizes the intermediate mild classes. The model ends up predicting either Neutral or an extreme class, skipping Mild Drop almost entirely: **F1[Mild Drop] = 0.000** (the class is never predicted). This pushes Macro F1 below the stratified dummy.

**When class weights work**: mild imbalance ratios (2:1 to 5:1), clean separable boundaries between classes. In a 5-class ordinal problem with 10:1 imbalance and overlapping distributions, it is too blunt an instrument.

---

#### Strategy 2 — RandomOverSampler (result: 0.250)

**What it does**: makes the training data balanced by randomly duplicating minority class samples until every class has the same count as the majority.

After oversampling (target: all classes match Neutral count ≈ 1871):

| Class | Before | After | Duplicates added |
| --- | --- | --- | --- |
| Strong Drop | 231 | 1871 | 1640 exact copies |
| Mild Drop | 594 | 1871 | 1277 exact copies |
| Strong Rise | 180 | 1871 | 1691 exact copies |

Total training set: from ≈2960 → ≈9355 samples (of which 6608 are duplicates).

**Why duplication helps at all**: even exact copies change the effective Gini weights. A node that splits 231 Strong Drop samples in half reduces Gini a small amount; a node that splits 1871 identical Strong Drop samples in half reduces Gini much more. The RF now devotes more splits to minority class regions.

**Why it is limited**: the 1640 duplicated Strong Drop samples are all identical to the original 231. The classifier can perfectly memorize these 231 feature vectors. But on the test set, any Strong Drop day that looks slightly different from a previously-seen crash day is not recognized. There is no generalization beyond the convex hull of observed minority samples. F1[Strong Drop] improves from 0.063 (Baseline) to 0.124 — doubling but still very low.

---

#### Strategy 3 — SMOTE (result: 0.289 — winner)

**What it does**: generates *synthetic* minority samples rather than duplicates, by interpolating between existing minority samples in feature space.

**The algorithm in full**:

1. For each minority class sample $x_i$:
   a. Find its $k$ nearest neighbors **within the same minority class only** (not the full dataset).
   b. Randomly select one neighbor $x_{nn}$ from the $k$ candidates.
   c. Draw $\lambda \sim U[0, 1]$.
   d. Create: $x_{new} = x_i + \lambda \cdot (x_{nn} - x_i)$

2. Repeat until the target class ratio is reached.

**Geometric interpretation**: $x_{new}$ lies on the line segment connecting $x_i$ and its neighbor $x_{nn}$. With $\lambda = 0$ the new point equals $x_i$; with $\lambda = 1$ it equals $x_{nn}$; in between, it is a blend. By sampling $\lambda$ uniformly, SMOTE distributes new points uniformly along all line segments inside the minority class.

```text
   x_i ─────────────────────────── x_nn
       ↑         ↑         ↑
     λ=0.1     λ=0.5     λ=0.9
    (near x_i)          (near x_nn)
    new synthetic points
```

**Why neighbors are searched in the minority class only**: if SMOTE searched all neighbors, it might interpolate between a Strong Drop sample and a Neutral sample, generating a synthetic point in the "middle" of the feature space between the two classes — exactly where no real class lives. By restricting to same-class neighbors, synthetic points are generated within the minority class manifold.

**The `safe_k_neighbors` guard**: in TimeSeriesSplit cross-validation, some folds contain very few samples of the rarest class. If Strong Rise has only 3 samples in a training fold, k=5 would crash (cannot find 5 neighbors in a set of 3). `safe_k_neighbors(y) = max(1, min(5, n_min - 1))` dynamically sets k to a safe value.

**Why SMOTE outperforms RandomOverSampler**: instead of memorizing 231 crash days, the classifier learns what a "crash day feature vector" looks like across a broader space of interpolated possibilities. On a test day that doesn't exactly match any historical crash, but whose feature vector is plausibly similar, the classifier has a better chance of recognizing it. F1[Strong Drop] = 0.237 vs 0.124 for RandomOverSampler.

**Why SMOTE's interpolation assumption is approximately valid here**: financial technical indicators (MACD, RSI, Bollinger) are continuous and slowly varying. A day halfway between two historical crash days in feature space (half the MACD magnitude, average RSI) is a plausible financial state — it does not violate any physical constraint. This is different from, say, categorical or binary features where interpolation produces meaningless intermediate values.

---

#### Strategy 4 — ADASYN (result: 0.280)

**What it does**: extends SMOTE with adaptive density — it generates *more* synthetic samples near minority class samples that are surrounded by majority class neighbors (harder to classify) and *fewer* near minority samples that are well-separated (easier to classify).

**The algorithm**:

1. For each minority sample $x_i$, find its $k$ nearest neighbors across the full dataset (majority + minority).
2. Count $\Delta_i$ = number of those neighbors belonging to the majority class.
3. Compute normalized difficulty: $\hat{r}_i = \Delta_i / k$ (proportion of majority neighbors).
4. Normalize to a distribution: $\hat{h}_i = \hat{r}_i \,/\, \sum_i \hat{r}_i$ (so $\sum_i \hat{h}_i = 1$).
5. Generate $G \cdot \hat{h}_i$ synthetic samples around $x_i$ (where $G$ is the total number of samples to generate). Each synthetic sample is generated the same way as SMOTE.

**What this achieves**: boundary minority samples (surrounded by many majority neighbors) get more synthetic support — the classifier is forced to build a better decision boundary in the hard regions. Interior minority samples (few majority neighbors) get fewer synthetics — they are already well-learned.

**Why ADASYN ≈ SMOTE here (0.280 vs 0.289)**: financial return classes have heavily overlapping, continuously blending distributions. There is no clean "interior" vs "boundary" for Strong Drop — the entire Strong Drop distribution overlaps with Mild Drop and Neutral. ADASYN's adaptive density doesn't find a clear easy-vs-hard distinction because the class boundaries are diffuse across all regions. The two methods end up generating similar synthetics in similar places.

**ADASYN's specific trade-off**: slightly better Strong Rise recall (0.303 vs 0.263) but slightly worse Strong Drop recall (0.204 vs 0.237) than SMOTE. This suggests ADASYN's density allocation happened to favour the Strong Rise boundary more, which is a stochastic property of the specific fold rather than a systematic advantage.

---

#### Strategy 5 — RandomUnderSampler (result: 0.194 — second worst)

**What it does**: instead of adding minority samples, it removes majority samples. The target is that all classes have the same count as the rarest class (Strong Rise, $n = 180$).

After undersampling:

| Class | Before | After | Samples discarded |
| --- | --- | --- | --- |
| Neutral | 1871 | 180 | **1691 discarded** |
| Mild Rise | 748 | 180 | 568 discarded |
| Mild Drop | 594 | 180 | 414 discarded |
| Strong Drop | 231 | 180 | 51 discarded |
| Strong Rise | 180 | 180 | 0 |

Total training set: ≈2960 → **900 samples** (70% reduction).

**Why this fails with 3,700 samples**: losing 70% of training data is catastrophic at this scale. Neutral F1 collapses to 0.246 — the classifier no longer has enough examples of what constitutes a Neutral day and confuses it with mild classes. Macro F1 = 0.194 falls below the stratified dummy.

**When undersampling works**: datasets with millions of majority class samples where the bottleneck is not information but computation. Reducing from 1,000,000 to 100,000 Neutral samples still provides abundant training signal while making the problem tractable and balanced.

---

#### Strategy 6 — SMOTEENN (result: 0.231 — at baseline level)

**What it does**: a two-step pipeline combining oversampling and cleaning.

**Step 1 — SMOTE**: oversample all minority classes to the majority class count (same as plain SMOTE).

**Step 2 — ENN (Edited Nearest Neighbors)**: for every sample in the now-oversampled training set (real or synthetic, majority or minority):

1. Find its $k$ nearest neighbors.
2. If the majority of those neighbors belong to a different class, **remove this sample**.

ENN removes:

- Synthetic minority samples that were placed too close to the majority class (generated in an ambiguous region)
- Real majority samples that sit deep inside a minority cluster (potential mislabels or boundary noise)

**Why this sounds good in theory**: after SMOTE, some synthetic Strong Drop points may lie near the Neutral cluster (because two Strong Drop training samples that were neighbors happened to be near the Neutral boundary). ENN removes these noisy synthetics, leaving a cleaner Strong Drop region for the classifier to learn from.

**Why it collapses in practice here**: in a 5-class ordinal problem with continuous, overlapping distributions, there is no clean boundary to clean. Many Neutral samples naturally sit near the Mild Drop / Mild Rise boundary — this is a property of the return distribution, not a labeling error. ENN removes these legitimate Neutral boundary samples because their nearest neighbors are Mild class samples. The Neutral training data is gutted: **F1[Neutral] = 0.274** (vs 0.546 for SMOTE). The loss in Neutral precision dominates the Macro F1 calculation, sinking SMOTEENN to the same level as the Baseline despite better minority class recall.

**ENN is most effective in**: binary classification with a clean, well-separated decision boundary where boundary noise genuinely hurts. In 5-class ordinal classification with fuzzy, adjacent-class overlap, ENN removes real signal rather than noise.

---

### Step 5 — The imblearn Pipeline and the resampling leakage problem

This is a subtle but critical implementation detail.

**The wrong way** (leaky):

```python
# WRONG: SMOTE is applied before cross-validation
X_train_smoted, y_train_smoted = SMOTE().fit_resample(X_train, y_train)
cross_val_score(rf, X_train_smoted, y_train_smoted, cv=TimeSeriesSplit(5))
```

The problem: SMOTE generates synthetic samples by interpolating between pairs of training points. When cross-validation then splits the data, the *original* training points that were used to generate synthetics may end up in the validation fold. The synthetic points in the training fold were derived from those validation points — the model effectively trained on information from its own validation set.

**The correct way** (imblearn.Pipeline):

```python
pipe = ImbPipeline([
    ('sampler', SMOTE(random_state=42, k_neighbors=k)),
    ('scaler', StandardScaler()),
    ('clf',    RandomForestClassifier(n_estimators=300)),
])
cross_val_score(pipe, X_train, y_train, cv=TimeSeriesSplit(5))
```

`imblearn.Pipeline` overrides the standard sklearn behavior: during `fit()` it applies the sampler to the training fold only. During `predict()` and on the validation fold, the sampler is not applied — the validation set sees only the original, real, non-synthetic samples. Each CV fold gets its own fresh SMOTE run on its own training data.

---

### Step 6 — TimeSeriesSplit: why KFold would be wrong

Standard `KFold` with `shuffle=True` randomly shuffles and splits the data. For temporal data with lagged features, this creates leakage at the feature level.

**Concrete example**: suppose day $t$ lands in the validation fold and day $t+1$ lands in the training fold.

- `nasdaq_lr_lag1` for day $t+1$ = the log return on day $t$ = the **target** for day $t$
- The classifier training on day $t+1$ directly observes what we are trying to predict for day $t$

`TimeSeriesSplit(n_splits=5)` uses expanding windows where every validation fold is temporally posterior to its training fold:

```text
Fold 1: train = [t_0 .. t_20%]      val = [t_20% .. t_40%]
Fold 2: train = [t_0 .. t_40%]      val = [t_40% .. t_60%]
Fold 3: train = [t_0 .. t_60%]      val = [t_60% .. t_80%]
Fold 4: train = [t_0 .. t_80%]      val = [t_80% .. t_100%]
```

No future information is ever in a training fold. This preserves the causal ordering that the model will face in deployment.

---

### Step 7 — Hyperparameter tuning (GridSearchCV)

GridSearch was run on the **SMOTEENN** pipeline (not SMOTE, the actual winner) searching:

```python
param_grid = {
    'clf__n_estimators':     [200, 400],
    'clf__max_depth':        [None, 20],
    'clf__min_samples_leaf': [1, 5],
}
cv = TimeSeriesSplit(n_splits=5)
```

The tuned model (Macro F1 = 0.224) underperforms untuned SMOTE (0.289) for two reasons:

1. **Wrong base strategy**: tuning was applied to SMOTEENN, which has a structural Neutral-collapse problem that no hyperparameter adjustment can fix.
2. **CV folds are noisier than the final holdout**: with ≈3,700 samples and 5 folds, each training fold has only ≈2,220 samples. The best hyperparameters found on these small folds do not necessarily generalize to training on the full 80% split. This is a well-known limitation of walk-forward CV on small financial datasets.

`class_weight` was deliberately excluded from the grid: SMOTEENN already rebalances the class distribution. Adding class weights on top would double-penalize the majority class, over-correcting in the opposite direction.

---

### Actual performance results

| Strategy | Macro F1 | F1[0 Drop] | F1[1 MildD] | F1[2 Neutral] | F1[3 MildR] | F1[4 Rise] | Bal. Acc | κ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **SMOTE** | **0.289** | 0.237 | 0.087 | 0.546 | 0.312 | 0.263 | 0.319 | 0.158 |
| ADASYN | 0.280 | 0.204 | 0.114 | 0.522 | 0.259 | 0.303 | 0.311 | 0.139 |
| RandomOverSampler | 0.250 | 0.124 | 0.048 | 0.551 | 0.254 | 0.274 | 0.272 | 0.106 |
| Baseline (no handling) | 0.231 | 0.063 | 0.070 | 0.559 | 0.212 | 0.252 | 0.256 | 0.089 |
| SMOTEENN | 0.231 | 0.175 | 0.179 | 0.274 | 0.228 | 0.298 | 0.273 | 0.064 |
| RandomUnderSampler | 0.194 | 0.130 | 0.129 | 0.246 | 0.191 | 0.272 | 0.246 | 0.042 |
| Class Weights | 0.187 | 0.085 | 0.000 | 0.562 | 0.144 | 0.143 | 0.231 | 0.055 |
| GridSearchCV Tuned | 0.224 | — | — | — | — | — | 0.263 | 0.058 |

**Winner: SMOTE (Macro F1 = 0.289).**

**Reading the table column by column**:

- **F1[0 Strong Drop]**: ranges from 0.063 (Baseline) to 0.237 (SMOTE). The biggest per-class gain from imbalance handling.
- **F1[1 Mild Drop]**: the hardest class for every strategy. Peaks at 0.179 (SMOTEENN) — but SMOTEENN achieves this by sacrificing Neutral F1 catastrophically. SMOTE gets only 0.087 on this class.
- **F1[2 Neutral]**: high for Baseline (0.559) and Class Weights (0.562) — models biased toward predicting Neutral do well here. Collapses to 0.246–0.274 for undersampling and SMOTEENN.
- **F1[3 Mild Rise]**: moderate across strategies. SMOTE achieves the best (0.312).
- **F1[4 Strong Rise]**: comparable to Strong Drop. Best under ADASYN (0.303).
- **κ**: very low even for SMOTE (0.158). This is the honest assessment: after chance correction, the model has achieved only modest real discriminative power. The financial signal is weak.

---

### SHAP Explainability

### SHAP global feature importance findings

Ranked by mean |SHAP| across all 5 classes and all test samples:

1. **`nasdaq_lr_lag1`** — the most recent NASDAQ log return is the single most informative feature across all classes. Consistent with weak short-term momentum: a positive lag1 tilts the forecast slightly toward Mild Rise or Neutral; a negative lag1 tilts toward Mild Drop.

2. **`nasdaq_vol_5d`** — the 5-day rolling realized volatility is the second most important feature. High 5-day volatility increases the predicted probability of **both** Strong Drop and Strong Rise simultaneously, pulling probability mass away from Neutral and the mild classes. This reflects that elevated uncertainty increases the likelihood of extreme outcomes regardless of direction.

3. **`nasdaq_rsi_14`** — the RSI momentum oscillator. RSI near or below 30 (oversold) nudges predictions toward Strong Drop. RSI near or above 70 (overbought) nudges predictions toward Strong Rise. RSI at mid-range supports Neutral.

4. **`nasdaq_macd_hist`** — momentum acceleration. A positive and growing histogram supports upward momentum (Mild/Strong Rise); a negative and declining histogram supports downward momentum (Mild/Strong Drop). This is the feature where MICE imputation quality matters most for downstream classification — mean imputation replaces it with a constant, destroying its discriminative signal.

5. **Macro and FX variables** (`us_rates_%`, `CPI`, `GDP`, `eur_usd`, `usd_chf`) — consistent with the Assignment 1 SARIMAX result: macro variables have statistically weak but non-zero signal at daily frequency. They appear in the lower half of SHAP importance, not the top.

6. **Other asset returns** (`sp500_log_return`, `gold_log_return`) — cross-market context provides supplementary signal that helps the classifier when NASDAQ-specific indicators are ambiguous.

### SHAP per-class findings (from beeswarm analysis)

**Class 0 — Strong Drop**:

- High `nasdaq_vol_5d` (red dots → right of zero): elevated volatility *increases* the probability of a Strong Drop prediction. Turbulent regimes produce more extreme negative moves.
- High `nasdaq_lr_lag1` (red dots → left of zero): strong positive recent momentum *reduces* the probability of a Strong Drop — the market just went up, making a crash less likely in the immediate short term.
- Low `nasdaq_rsi_14` (blue dots → right of zero): oversold conditions contribute to Strong Drop predictions, consistent with momentum continuation in downtrends.

**Class 4 — Strong Rise**:

- High `nasdaq_vol_5d` (red dots → right of zero): high volatility increases Strong Rise probability — the same elevated uncertainty that makes Strong Drop more likely also makes Strong Rise more likely. Volatility is symmetric in its effect on extreme outcomes.
- High `nasdaq_lr_lag1` (red dots → right of zero): positive recent momentum slightly increases Strong Rise probability — momentum continuation.

**Symmetry and asymmetry**: if markets were perfectly symmetric, the Strong Drop and Strong Rise beeswarm panels would be mirror images. The actual divergence in feature rankings between the two panels reveals financial market asymmetries — crashes tend to be sharper, faster, and driven by different psychological dynamics (fear, deleveraging) than rallies (gradual buying, FOMO). The SHAP analysis quantifies this asymmetry by class.

### Confusion matrix error patterns

The dominant error pattern across all strategies (from confusion matrix analysis):

- **True Strong Drop (class 0)** → most often predicted as Neutral or Mild Drop. The model rarely produces a Strong Drop prediction, even when the true outcome is extreme.
- **True Strong Rise (class 4)** → most often predicted as Neutral or Mild Rise. Same failure mode.
- **True Neutral (class 2)** → correctly predicted in most cases (diagonal ≈0.55–0.60), as this is the class the model is implicitly biased toward.
- **True Mild Drop (class 1)** → the most confused class, spread across classes 0, 1, 2. Mild Drop sits between Strong Drop and Neutral with blurry boundaries.

**SMOTE vs Baseline in the confusion matrix**: SMOTE's improvement is visible in the Strong Drop row — the diagonal entry rises from ≈0.04 (Baseline) to ≈0.16 (SMOTE). The Neutral diagonal is preserved (≈0.54 vs ≈0.57). SMOTEENN shows the opposite: Strong Drop diagonal improves to ≈0.12, but Neutral diagonal collapses to ≈0.28.

**Financial cost asymmetry**: missing a Strong Drop (predicting Neutral when a crash is coming) is far more costly than a false alarm. The confusion matrix pattern reveals that the model is systematically biased toward under-predicting extreme events — a conservative bias that limits its usefulness for tail-risk management.

---

## Issues and Discrepancies Found

### 1. Documentation claims SMOTEENN wins — SMOTE is the actual winner

The Notebook 03 conclusions cell and the original Memorie.md both state: *"SMOTEENN produces the best overall Macro F1."* This is **factually incorrect** based on the actual results.

From `03_strategy_comparison.csv`:

- SMOTE Macro F1 = **0.2892** (winner)
- SMOTEENN Macro F1 = 0.2307 (barely above baseline of 0.2313)

The `performance_summary.csv` also records: "Best strategy (SMOTE)". The documentation narrative needs to be corrected everywhere it claims SMOTEENN is the winner.

**Why did SMOTEENN underperform?** SMOTEENN's ENN cleaning step removes boundary-region samples from both classes. In a 5-class problem with overlapping class boundaries (financial return classes 1, 2, 3 heavily overlap), ENN cleaning may remove too many samples near the boundaries, collapsing the F1 for Neutral (0.274 vs 0.546 for SMOTE). SMOTEENN trades Neutral class quality for slightly better Strong Drop/Rise recall, but the net effect on Macro F1 is negative.

### 2. GridSearchCV tuned model (0.2245) is worse than untuned SMOTE (0.2892)

The tuning was applied to SMOTEENN specifically, not to SMOTE (the actual winner). This is a methodological inconsistency: the pipeline that wins the strategy comparison (SMOTE) was not the one that received hyperparameter optimization.

Additionally, `TimeSeriesSplit` CV validation sets are smaller and from an earlier time period than the final test set — the best hyperparameters found by CV on the SMOTEENN pipeline may not generalize to the final holdout. This gap between tuned CV performance and final test performance is a known limitation of time-series cross-validation.

**Recommendation for future work**: run `GridSearchCV` on the SMOTE pipeline (the winner), not on SMOTEENN.

### 3. Class Weights (0.187) underperforms Dummy (stratified) (0.210)

The Class Weights strategy produced F1 = 0.000 for Mild Drop (class 1) and relatively low scores for Mild Rise (class 3). The model appears to have collapsed into predicting mostly Neutral and the two extreme classes, skipping the intermediate mild classes entirely. This can happen when class weights create large asymmetric loss penalties that push the decision boundary through the mild classes without stabilizing a clear region for them. It is the worst-performing non-trivial strategy in this experiment.

---

## Key Technical Concepts

### Why Macro F1 is the primary metric

**Accuracy** rewards correct Neutral predictions disproportionately (52% of the data). A model that always predicts Neutral scores 52% accuracy but is economically useless.

**Macro F1** computes F1 independently for each class and averages uniformly:
$$\text{Macro F1} = \frac{1}{K}\sum_{k=0}^{K-1} \frac{2 \cdot P_k \cdot R_k}{P_k + R_k}$$

This gives equal weight to Strong Drop (6%) and Neutral (52%), making it sensitive to minority class performance. A model that gains on minority recall at the cost of Neutral precision may or may not improve Macro F1 — the trade-off is explicit.

**Balanced Accuracy** = average of per-class recall = `(1/K) Σ R_k`. Equivalent to Macro Recall.

**Cohen's κ** corrects Macro Accuracy for chance agreement, giving 0 for random and 1 for perfect. Useful for comparing across datasets with different imbalance ratios.

### Why TimeSeriesSplit is mandatory for CV

`KFold` with `shuffle=True` randomly interleaves train and validation data. For a feature like `nasdaq_lr_lag1` (yesterday's return), a shuffled split may place day $t$ in validation and day $t-1$ in training — the lag then sees future information at training time. `TimeSeriesSplit` guarantees that every validation fold is entirely posterior to its training fold, preserving the causal ordering.

### The resampling leakage problem

SMOTE must be applied **inside** the cross-validation loop, not outside. If you SMOTE the full training set before fitting the CV, the synthetic samples are interpolations of real samples that may appear in different CV folds — the validation fold effectively sees information from training samples it should not. `imblearn.Pipeline` solves this by applying the sampler only during `fit`, never during `predict` or on the held-out fold.

---

## Technical Stack

- `sklearn.impute.IterativeImputer` (experimental) — MICE
- `sklearn.impute.KNNImputer` — KNN imputation
- `imblearn.over_sampling.SMOTE`, `ADASYN`, `RandomOverSampler`
- `imblearn.under_sampling.RandomUnderSampler`
- `imblearn.combine.SMOTEENN`
- `imblearn.pipeline.Pipeline` — ensures resampling stays inside the CV loop
- `sklearn.model_selection.TimeSeriesSplit` + `GridSearchCV`
- `shap.TreeExplainer` — exact Shapley values for tree models
- Metrics: `f1_score(average='macro')`, `balanced_accuracy_score`, `cohen_kappa_score`
