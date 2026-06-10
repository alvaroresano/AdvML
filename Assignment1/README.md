# Advanced Financial Time Series Forecasting

> A rigorous 6-phase quantitative pipeline combining statistical preprocessing, classical econometric models, volatility modelling, a PatchTST-style transformer, and a walk-forward financial backtest — all applied to multivariate financial market data spanning 2010–2024.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Dataset](#3-dataset)
4. [Phase 1 — Data Engineering & Statistical Preparation](#4-phase-1--data-engineering--statistical-preparation)
5. [Phase 2 — STL Decomposition](#5-phase-2--stl-decomposition)
6. [Phase 3 — Classical Forecasting Baseline (SARIMAX)](#6-phase-3--classical-forecasting-baseline-sarimax)
7. [Phase 4 — Volatility Modelling (GARCH)](#7-phase-4--volatility-modelling-garch)
8. [Phase 5 — Deep Forecasting (PatchTST-style Transformer)](#8-phase-5--deep-forecasting-patchtst-style-transformer)
9. [Phase 6 — Walk-Forward Backtesting](#9-phase-6--walk-forward-backtesting)
10. [Key Results Summary](#10-key-results-summary)
11. [How to Run](#11-how-to-run)
12. [Dependencies](#12-dependencies)

---

## 1. Project Overview

This project studies multivariate financial time series forecasting using historical market and macroeconomic data covering 7 assets and 3 macro indicators from April 2010 to October 2024. The objective is not only to produce accurate forecasts but to build a **rigorous quantitative pipeline** that progresses through:

- statistically sound preprocessing with formal stationarity testing,
- interpretable classical time series models as a defensible baseline,
- volatility-aware risk modelling (GARCH),
- modern transformer-based deep learning (PatchTST-style architecture),
- and realistic financial backtesting under explicit market frictions.

The pipeline is structured as **six sequential phases**, each fully documented and producing persisted outputs that the next phase consumes. Every modeling decision is grounded in mathematical rationale and empirical findings from the previous phase.

**Primary forecasting target:** NASDAQ-100 next-day log return.

---

## 2. Repository Structure

```
Assignment1/
├── start.ipynb                          # Entry-point notebook
├── 01_EDA.ipynb                         # Exploratory data analysis
├── 03_Forecasting_LSTM_&_Chronos.ipynb  # LSTM / Chronos exploration
├── VisualAnalytics.ipynb                # Full visualisation notebook
├── VisualAnalytics.html                 # Static HTML export of visualisations
├── financial_regression.csv            # Raw dataset (Kaggle)
├── docs/
│   └── Unit 4 - AML.pdf
├── images/                             # EDA plots
├── scripts/
│   ├── run_phase1.py                   # Phase 1 pipeline runner
│   ├── run_phase2.py
│   ├── run_phase3.py
│   ├── run_phase4.py
│   ├── run_phase5.py
│   ├── run_phase6.py
│   └── generate_visualization_notebook.py
├── src/
│   └── advml_assignment1/
│       ├── __init__.py
│       ├── phase1_data_engineering.py
│       ├── phase2_stl_decomposition.py
│       ├── phase3_classical_baseline.py
│       ├── phase4_volatility_modeling.py
│       ├── phase5_deep_forecasting.py
│       └── phase6_backtesting.py
└── outputs/
    ├── phase1/    — cleaned_data.csv, featured_data.csv, modeling_data.csv, adf_summary.csv
    ├── phase2/    — stl_decomposition_components.csv, stl_decomposition_summary.csv, plots/
    ├── phase3/    — model_metadata.json, coefficient_summary.csv, test_forecasts.csv,
    │               residual_diagnostics.csv, qq_plot_data.csv, train_fitted.csv
    ├── phase4/    — garch_parameter_summary.csv, garch_model_metadata.json, garch_residual_diagnostics.csv,
    │               train_volatility.csv, test_volatility_forecasts.csv, combined_mean_residuals.csv
    ├── phase5/    — patchtst_state_dict.pt, model_metadata.json, training_history.csv,
    │               validation_predictions.csv, test_predictions.csv, feature_schema.csv, phase5_design_data.csv
    └── phase6/    — fold_definitions.csv, fold_metrics.csv, model_metadata.json,
                    prediction_records.csv, strategy_daily_returns.csv, strategy_summary.csv
```

---

## 3. Dataset

**Source:** [Financial Data — Kaggle](https://www.kaggle.com/datasets/franciscogcc/financial-data)

| Asset / Variable | Type | Frequency |
|---|---|---|
| NASDAQ-100 | Equity index | Daily |
| S&P 500 | Equity index | Daily |
| Gold | Commodity | Daily |
| Oil (WTI) | Commodity | Daily |
| Silver | Commodity | Daily |
| Platinum | Commodity | Daily |
| Palladium | Commodity | Daily |
| EUR/USD | FX rate | Daily |
| USD/CHF | FX rate | Daily |
| GDP | Macroeconomic | Quarterly |
| CPI | Macroeconomic | Monthly |
| US Federal Funds Rate | Macroeconomic | Monthly (meeting-driven) |

**Sample period:** 2010-04-01 to 2024-10-18 (~3,700 trading days after preprocessing).

---

## 4. Phase 1 — Data Engineering & Statistical Preparation

**Module:** `src/advml_assignment1/phase1_data_engineering.py`

### Mixed-Frequency Identification

The dataset is a mixed-frequency panel. Phase 1 formally identifies the update frequency of each variable by computing inter-observation gaps:
- GDP: ~91-day mean gap → quarterly
- CPI: ~30-day mean gap → monthly
- US Rates: ~30-day mean gap → monthly (FOMC meeting-driven)

Macro variables are **forward-filled** after trimming the sample to 2010-04-01 (the first non-null GDP date). This implements the "last known value" interpretation — the only defensible strategy when the series is published discretely rather than interpolated daily.

### Target Transformation: Log Returns

Raw price levels are non-stationary (trending, non-constant variance). Log returns are used throughout the project for three mathematically rigorous reasons:

1. **Time-additivity:** $r_{t_1 \to t_3} = r_{t_1 \to t_2} + r_{t_2 \to t_3}$ — log returns sum across periods, enabling linear time-series modelling.
2. **Symmetry:** gains and losses are symmetric ($\log(2) = -\log(0.5)$), unlike simple returns.
3. **Domain consistency:** $\log(P_t/P_{t-1}) \in (-\infty, +\infty)$, matching the support of Gaussian and Student-t error distributions.

The deeper theoretical justification comes from **Geometric Brownian Motion (GBM)** and Itô's lemma, which formally shows that log returns are Gaussian under GBM with mean $(\mu - \sigma^2/2)T$ and variance $\sigma^2 T$.

### Stationarity Testing: ADF

The Augmented Dickey-Fuller test is applied to all return series. All log return series are confirmed stationary (unit root rejected), so $d = 0$ in subsequent ARIMA modelling — no additional differencing is required.

### Technical Indicators

RSI(14), MACD (12/26/9), and Bollinger z-score (20-day) are computed for all 7 assets and stored in the modeling dataset for Phase 5.

**Outputs:** `outputs/phase1/cleaned_data.csv`, `featured_data.csv`, `modeling_data.csv`, `adf_summary.csv`

---

## 5. Phase 2 — STL Decomposition

**Module:** `src/advml_assignment1/phase2_stl_decomposition.py`

STL (Seasonal-Trend decomposition using Loess) decomposes each log-price series into additive trend, seasonal, and residual components. Because log prices are multiplicative processes, STL is applied to log prices rather than raw prices, making the decomposition additive:

$$\log P_t \approx \log T_t + \log S_t + \log E_t$$

**Key finding:** Seasonal strength ≈ 0.056 for NASDAQ — essentially negligible. There is no stable weekly seasonal pattern in daily returns. This directly informs the SARIMAX specification in Phase 3: seasonal terms are set to $(P, D, Q, m) = (0, 0, 0, 0)$.

**Outputs:** `outputs/phase2/` — decomposition components (CSV), STL plots per asset (PNG).

---

## 6. Phase 3 — Classical Forecasting Baseline (SARIMAX)

**Module:** `src/advml_assignment1/phase3_classical_baseline.py`

### Model Selection

`pmdarima.auto_arima` searches over candidate ARIMA orders using AIC (Akaike Information Criterion):

$$\text{AIC} = 2k - 2\log L$$

The selected order is then refit as a `statsmodels` SARIMAX model for full diagnostics and forecasting infrastructure.

**Selected specification:** SARIMAX(0,0,0)(0,0,0,0) with an intercept and 8 lagged exogenous predictors.

The (0,0,0) ARIMA order is a meaningful empirical result, not a failure. It means: after transforming the target into log returns and adding the lagged exogenous block, the data does not justify adding AR or MA terms to the conditional mean equation. This is consistent with the Efficient Market Hypothesis — simple linear patterns in NASDAQ returns are arbitraged away; the predictable structure lives in cross-asset linkages (the exogenous block) and in the variance process (GARCH, Phase 4).

### Exogenous Design Matrix

| Column | Variable | Why lagged by 1 day |
|---|---|---|
| `sp500_ret_l1` | S&P 500 log return | Same-day S&P 500 is unknown when forecasting NASDAQ open |
| `gold_ret_l1` | Gold log return | Leakage prevention: contemporaneous correlation would inflate accuracy |
| `oil_ret_l1` | WTI crude log return | Same rationale |
| `eur_usd_ret_l1` | EUR/USD log return | FX close is contemporaneous with NASDAQ |
| `usd_chf_ret_l1` | USD/CHF log return | Same |
| `gdp_growth_l1` | Log GDP growth rate | Quarterly release: only yesterday's reading is available |
| `cpi_inflation_l1` | Log CPI inflation rate | Monthly release |
| `rate_change_l1` | Simple change in US funds rate | Percentage-point change (not log, because rate can be near zero) |

### Test-Set Results

| Metric | Value |
|---|---|
| Test RMSE | 0.01103 |
| Test MAE | 0.00824 |
| Directional accuracy | 52.78% |
| Test period | 2023-10-17 to 2024-10-18 (252 obs) |

**Residual diagnostics:** Ljung-Box strongly rejects white-noise residuals at lags 10 and 20; Jarque-Bera strongly rejects normality; Q-Q plot shows heavy-tailed departures. This motivates Phase 4 (volatility modelling) — the conditional mean is partially explained, but substantial serial dependence and non-normality remain in squared residuals.

**Outputs:** `outputs/phase3/`

---

## 7. Phase 4 — Volatility Modelling (GARCH)

**Module:** `src/advml_assignment1/phase4_volatility_modeling.py`

### Division of Labour: SARIMAX + GARCH

SARIMAX models the **conditional mean** $E[y_t | \mathcal{F}_{t-1}]$. GARCH models the **conditional variance** $\text{Var}[y_t | \mathcal{F}_{t-1}] = \sigma_t^2$. Together they describe the full conditional distribution:

$$y_t \mid \mathcal{F}_{t-1} \sim \text{scaled-}t_\nu\!\left(\hat{y}_t^{\text{SARIMAX}},\; \sigma_t^2\right)$$

GARCH is fit on the Phase 3 SARIMAX residuals. A **Student-t innovation distribution** is used because Phase 3 diagnostics already confirmed heavy tails.

### GARCH(1,1) Specification

$$\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

| Parameter | Estimated Value | Interpretation |
|---|---|---|
| $\omega$ | 0.02733 | Baseline variance floor |
| $\alpha$ | 0.12767 | Shock reactivity: a large past shock raises today's variance by ~13% of its magnitude |
| $\beta$ | 0.86396 | Variance persistence: 86% of yesterday's variance carries into today |
| $\nu$ | 5.8907 | Student-t degrees of freedom — materially heavier tails than Gaussian |

**Derived quantities:**

| Quantity | Value | Meaning |
|---|---|---|
| Persistence ($\alpha + \beta$) | 0.9916 | Volatility shocks decay very slowly |
| Unconditional volatility | ~1.81%/day | Long-run daily return standard deviation |
| Volatility half-life | ~82 trading days | A volatility shock takes ~4 calendar months to decay by half |

**Key diagnostic improvement:** After GARCH filtering, the Ljung-Box test on **squared** standardised residuals is no longer significant — the volatility clustering is substantially absorbed, even though some linear structure remains in the standardised residuals themselves.

The GARCH conditional variance $\sigma_t^2$ is passed as an input feature to the Phase 5 LSTM, allowing the deep model to condition its predictions on the current volatility regime.

**Outputs:** `outputs/phase4/`

---

## 8. Phase 5 — Deep Forecasting (PatchTST-style Transformer)

**Module:** `src/advml_assignment1/phase5_deep_forecasting.py`

### Architecture: PatchTST-style Multivariate Transformer

Inspired by the PatchTST paper (*"A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"*, Nie et al.), this model addresses the limitations of linear SARIMAX by learning non-linear interactions across features and temporal patches.

**Core idea:** instead of treating each daily observation as an individual token, the model groups consecutive days into **overlapping temporal patches** and processes those patches with a Transformer encoder.

### Tensor Flow

| Stage | Shape | Operation |
|---|---|---|
| Input | [B, 60, 33] | Lookback window: 60 days, 33 features |
| Transpose | [B, 33, 60] | Channel-first for patching |
| Patch extraction | [B, 33, 11, 10] | 11 overlapping patches of 10 days each (stride=5) |
| Patch embedding + positional | [B, 33, 11, 32] | Linear projection to d_model=32 |
| Reshape for encoder | [B×33, 11, 32] | Channel-independent view |
| Transformer encoder | [B×33, 11, 32] | 2 layers, 4 attention heads |
| Mean pool + reshape | [B, 33, 32] | One representation vector per channel |
| Flatten | [B, 1056] | All 33 channels concatenated |
| MLP head | [B, 1] | Next-day NASDAQ log return forecast |

**Number of patches:** $N_{\text{patches}} = 1 + \frac{L - P}{S} = 1 + \frac{60 - 10}{5} = 11$

**Channel independence** is implemented by merging batch and channel dimensions before the Transformer encoder, so attention is computed independently within each channel's own patch sequence — consistent with the PatchTST design principle.

### Input Features (33 total)

| Feature group | Count | Examples |
|---|---|---|
| Asset log returns | 7 | nasdaq, sp500, gold, oil, silver, platinum, palladium |
| RSI(14) | 7 | `nasdaq_rsi_14`, `sp500_rsi_14`, ... |
| MACD histogram | 7 | `nasdaq_macd_hist`, ... |
| Bollinger z-score | 7 | `nasdaq_bb_zscore`, ... |
| FX log returns (lag-1) | 2 | `eur_usd_ret_l1`, `usd_chf_ret_l1` |
| Macro changes (lag-1) | 3 | `gdp_growth_l1`, `cpi_inflation_l1`, `rate_change_l1` |

All inputs are lagged by 1 day relative to the target. No same-day information is ever used.

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Lookback window | 60 days |
| Patch length | 10 days |
| Patch stride | 5 days (overlapping) |
| d_model | 32 |
| Attention heads | 4 |
| Transformer layers | 2 |
| Feedforward dim | 64 |
| Dropout | 0.10 |
| Optimizer | AdamW (lr=0.001, wd=1e-4) |
| Batch size | 64 |
| Early stopping patience | 8 epochs (best: epoch 9) |

### Test-Set Results (Fixed Holdout: Oct 2023 – Oct 2024)

| Metric | PatchTST | SARIMAX Baseline | Improvement |
|---|---|---|---|
| RMSE | 0.010898 | 0.011026 | −1.2% |
| MAE | 0.008096 | 0.008240 | −1.7% |
| Directional accuracy | 57.14% | 52.78% | +4.4 pp |
| Forecast/actual correlation | 0.1269 | — | — |

The deep model improves all three holdout metrics, but the improvement is modest — consistent with the low signal-to-noise ratio inherent to daily financial return forecasting. Directional accuracy of 57.1% is economically relevant even if the magnitude forecasts remain conservative (compressed near zero due to MSE training).

**Outputs:** `outputs/phase5/`

---

## 9. Phase 6 — Walk-Forward Backtesting

**Module:** `src/advml_assignment1/phase6_backtesting.py`

### Motivation

A single holdout split tells us how models performed on one historical segment. It does not test stability across multiple market regimes, does not account for repeated retraining drift, and does not answer whether the forecast skill is economically useful once trading costs are included. Phase 6 addresses all three.

### Rolling-Window Design

| Parameter | Value |
|---|---|
| Training window | 2,000 observations |
| Validation window | 252 observations (1 year) |
| Test window | 252 observations (1 year) |
| Step size | 252 observations |
| Total folds | 5 (non-overlapping test windows) |

Both models (SARIMAX and PatchTST) are **fully retrained from scratch** in every fold.

### Trading Rule & Market Frictions

$$\text{position}_t = \text{sign}(\hat{r}_t) \in \{-1, 0, +1\}$$

Long on positive forecast, short on negative forecast. Costs are applied on every position change:

| Cost component | Value |
|---|---|
| Commission | 2 basis points |
| Slippage | 3 basis points |
| Total cost per unit turnover | 5 basis points |

$$R_t^{\text{net}} = \text{position}_t \cdot r_t - \text{turnover}_t \cdot 0.0005$$

### Rolling Forecast Metrics (Aggregate Across 5 Folds)

| Metric | SARIMAX | PatchTST |
|---|---|---|
| RMSE | 0.016122 | 0.015967 |
| MAE | 0.011522 | 0.011231 |
| Directional accuracy | 51.83% | 53.49% |

### Trading Results After Costs

| KPI | SARIMAX | PatchTST |
|---|---|---|
| Average turnover | 0.6865 | 0.2262 |
| Gross cumulative return | 155.64% | 83.94% |
| **Net cumulative return** | **65.81%** | **59.47%** |
| **Annualised Sharpe ratio** | **0.5251** | **0.4955** |
| Maximum drawdown | −40.42% | −42.59% |

### The Core Financial ML Lesson

PatchTST wins on **all statistical metrics** (RMSE, MAE, directional accuracy) but SARIMAX wins on **all financial metrics** (Sharpe ratio, net return, max drawdown). This discrepancy arises from four mechanisms:

1. **Hit rate weights all days equally; P&L does not.** Getting the sign right on a ±2% day earns 40× more than on a ±0.05% day.
2. **Forecast amplitude determines position stability.** PatchTST's MSE-optimal forecasts are compressed near zero and therefore flip sign more easily from small fluctuations, wasting trades on unimportant days. SARIMAX's larger-amplitude signals are more decisive and generate fewer spurious position changes.
3. **Turnover asymmetry.** Despite paying more per day in costs (3.4 bps vs 1.1 bps), SARIMAX's signals are *worth paying for* — the incremental gross P&L from acting on them exceeds the extra cost.
4. **Regime dependence.** Neither model consistently dominates across all 5 folds. The aggregate winner depends on which model happened to be better during the highest-return market regimes.

This result is a canonical example of why financial ML must evaluate **economic utility**, not only forecast error. Statistical superiority does not guarantee commercial advantage.

**Outputs:** `outputs/phase6/`

---

## 10. Key Results Summary

| Phase | Model | Primary Metric | Value |
|---|---|---|---|
| Phase 3 | SARIMAX(0,0,0) + 8 exogenous regressors | Directional accuracy (test) | 52.78% |
| Phase 3 | SARIMAX | RMSE (test) | 0.01103 |
| Phase 4 | GARCH(1,1)-t | Persistence ($\alpha+\beta$) | 0.9916 |
| Phase 4 | GARCH(1,1)-t | Volatility half-life | ~82 trading days |
| Phase 5 | PatchTST-style Transformer | Directional accuracy (test) | 57.14% |
| Phase 5 | PatchTST-style Transformer | RMSE (test) | 0.01090 |
| Phase 6 | SARIMAX (rolling backtest) | Annualised Sharpe | 0.5251 |
| Phase 6 | PatchTST (rolling backtest) | Annualised Sharpe | 0.4955 |

---

## 11. How to Run

Each phase can be run independently via its script, provided all upstream outputs are available.

```bash
cd Assignment1

# Phase 1: data engineering (must be run first)
python scripts/run_phase1.py

# Phase 2: STL decomposition
python scripts/run_phase2.py

# Phase 3: SARIMAX baseline
python scripts/run_phase3.py

# Phase 4: GARCH volatility model
python scripts/run_phase4.py

# Phase 5: PatchTST deep forecasting
python scripts/run_phase5.py

# Phase 6: rolling backtest
python scripts/run_phase6.py
```

Alternatively, run all phases through the entry-point notebook:

```bash
jupyter notebook start.ipynb
```

Visualisations are generated by:

```bash
jupyter notebook VisualAnalytics.ipynb
# or open the pre-rendered HTML version:
# open VisualAnalytics.html
```

---

## 12. Dependencies

See `../requirements.txt` for the full pinned dependency list. Core packages:

| Package | Version | Purpose |
|---|---|---|
| `pandas` | 2.2.3 | Data loading and manipulation |
| `numpy` | 2.4.4 | Numerical arrays |
| `statsmodels` | 0.14.5 | SARIMAX, Ljung-Box, ADF tests |
| `pmdarima` | 2.0.4 | `auto_arima` order selection |
| `arch` | 7.2.0 | GARCH(1,1) with Student-t innovations |
| `torch` | 2.5.1 | PatchTST-style transformer training |
| `scikit-learn` | 1.6.0 | Scalers, metrics, TimeSeriesSplit |
| `matplotlib` / `seaborn` / `plotly` | latest | Visualisations |
| `chronos-forecasting` | 2.2.2 | Foundation model reference (Chronos-2) |

---

*Course: Advanced Machine Learning — Assignment 1: Time Series and Forecasting*
