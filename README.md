# Advanced Machine Learning — Financial Markets Pipeline

> A production-style academic repository covering two complete machine learning projects applied to financial market data spanning 2010–2024. Built as coursework for the Advanced Machine Learning course.

---

## Repository Overview

This repository contains **two independent but connected projects**, both built on the same underlying financial dataset (NASDAQ-100, S&P 500, commodities, FX, and macroeconomic variables). Each assignment introduces a distinct set of technical challenges and is self-contained with its own notebooks, source modules, and outputs.

| Assignment | Topic | Core Challenges |
|---|---|---|
| [**Assignment 1 →**](#assignment-1--financial-time-series-forecasting) | Time Series Forecasting | Stationarity, SARIMAX, GARCH, PatchTST transformer, walk-forward backtesting |
| [**Assignment 2 →**](#assignment-2--data-imputation-imbalanced-classification--explainability) | Imputation, Classification & Explainability | MCAR missingness, MICE, SMOTE, imbalanced Random Forest, SHAP |

---

## Quick Navigation

```
AdvML-main/
├── 📁 Assignment1/       ← Time Series & Forecasting
├── 📁 Assignment2/       ← Imputation, Classification & Explainability
├── 📄 Project_Memoire.md ← Full 3,200-line master documentation
├── 📄 PRESENTATION.md    ← Complete oral presentation script
├── 📄 requirements.txt   ← Full pinned dependency list
└── 📁 memory/            ← Project context and notes
```

---

## Dataset

Both assignments share the same source dataset.

**Source:** [Financial Data — Kaggle](https://www.kaggle.com/datasets/franciscogcc/financial-data)

| Category | Variables | Frequency |
|---|---|---|
| Equity indices | NASDAQ-100, S&P 500 (OHLCV) | Daily |
| Commodities | Gold, Silver, Oil, Platinum, Palladium (OHLCV) | Daily |
| FX rates | EUR/USD, USD/CHF | Daily |
| Macroeconomic | GDP | Quarterly |
| Macroeconomic | CPI, US Federal Funds Rate | Monthly |

**Period:** April 2010 → October 2024 (~3,700 trading days after cleaning)

The dataset is a **mixed-frequency panel** — daily market prices alongside quarterly and monthly macro releases — which requires careful forward-fill alignment before any modeling.

---

## Assignment 1 — Financial Time Series Forecasting

📁 **[`Assignment1/`](./Assignment1/)**

**Goal:** Forecast the next-day NASDAQ-100 log return using a rigorous 6-phase quantitative pipeline that progresses from classical econometrics through deep learning, and evaluates the result under realistic walk-forward trading conditions.

### Pipeline Architecture

```
Raw CSV
  ↓
Phase 1 — Data Engineering & ADF Stationarity Tests
  ↓
Phase 2 — STL Decomposition (trend / seasonality / residual)
  ↓
Phase 3 — SARIMAX Classical Baseline (conditional mean)
  ↓
Phase 4 — GARCH(1,1)-t Volatility Model (conditional variance)
  ↓
Phase 5 — PatchTST-style Transformer (nonlinear mean forecasting)
  ↓
Phase 6 — 5-Fold Walk-Forward Backtest with Transaction Costs
```

### Key Results

| Phase | Model | Metric | Value |
|---|---|---|---|
| Phase 3 | SARIMAX(0,0,0) + 8 exogenous | Directional accuracy (test) | 52.8% |
| Phase 4 | GARCH(1,1) Student-t | Volatility persistence (α+β) | 0.9916 |
| Phase 4 | GARCH(1,1) Student-t | Volatility half-life | ~82 trading days |
| Phase 5 | PatchTST-style Transformer | Directional accuracy (test) | 57.1% |
| Phase 6 | SARIMAX (rolling backtest) | Net cumulative return | +65.8% |
| Phase 6 | PatchTST (rolling backtest) | Net cumulative return | +59.5% |
| Phase 6 | SARIMAX (rolling backtest) | Annualised Sharpe ratio | 0.525 |

**Headline finding:** PatchTST outperforms SARIMAX on every statistical metric (RMSE, MAE, directional accuracy) yet SARIMAX delivers better economic results in the backtest (higher Sharpe, lower drawdown). This discrepancy — a canonical result in financial ML — is fully explained in the Phase 6 documentation.

### Entry Points

| File | Purpose |
|---|---|
| [`start.ipynb`](./Assignment1/start.ipynb) | Entry-point notebook — run all phases sequentially |
| [`01_EDA.ipynb`](./Assignment1/01_EDA.ipynb) | Exploratory data analysis |
| [`03_Forecasting_LSTM_&_Chronos.ipynb`](./Assignment1/03_Forecasting_LSTM_%26_Chronos.ipynb) | Hybrid LSTM and Chronos-T5 exploration |
| [`VisualAnalytics.ipynb`](./Assignment1/VisualAnalytics.ipynb) | Interactive Plotly visualisation notebook |
| [`VisualAnalytics.html`](./Assignment1/VisualAnalytics.html) | Static HTML export (no kernel required) |
| [`scripts/run_phaseN.py`](./Assignment1/scripts/) | Run individual phases as standalone scripts |
| [`src/advml_assignment1/`](./Assignment1/src/advml_assignment1/) | Production-quality Python modules for all 6 phases |
| [`outputs/`](./Assignment1/outputs/) | Persisted CSVs, JSONs, model weights, and plots |

📖 **[Full Assignment 1 README →](./Assignment1/README.md)**

---

## Assignment 2 — Data Imputation, Imbalanced Classification & Explainability

📁 **[`Assignment2/`](./Assignment2/)**

**Goal:** Convert the regression target from Assignment 1 into a 5-class ordinal classification problem, benchmark four imputation strategies under controlled MCAR missingness, evaluate six imbalance-correction approaches under Macro F1, and explain the winning classifier with SHAP TreeExplainer.

### Three-Notebook Pipeline

```
Notebook 01 — EDA & Target Creation
    5-class label from NASDAQ log return thresholds
    Class distribution: Neutral ≈ 52%, Strong Drop ≈ 6%, Strong Rise ≈ 5%
  ↓
Notebook 02 — Imputation Methods
    5% MCAR injected into 8 technical indicator columns
    Benchmark: Mean vs Median vs KNN vs MICE
    Winner: MICE (16× improvement on MACD reconstruction)
  ↓
Notebook 03 — Imbalanced Classification & Explainability
    6 strategies: Class Weights / RandomOverSampler / SMOTE /
                  ADASYN / RandomUnderSampler / SMOTEENN
    Winner: SMOTE (Macro F1 = 0.289)
    Explainability: SHAP TreeExplainer on winning RF pipeline
```

### Key Results

| Task | Method | Metric | Value |
|---|---|---|---|
| Imputation | MICE | MAE on MACD (vs Mean = 4.798) | **0.294 (16× better)** |
| Imputation | MICE | MAE on MACD histogram (vs Mean = 1.261) | **0.089 (14× better)** |
| Classification floor | Stratified dummy | Macro F1 | 0.210 |
| Classification | SMOTE + Random Forest | Macro F1 | **0.289** |
| Classification | SMOTE + Random Forest | F1[Strong Drop] (vs Baseline = 0.063) | **0.237** |
| SHAP | Top feature | Global importance rank | `nasdaq_lr_lag1` (#1) |
| SHAP | Volatility feature | Effect on extreme classes | Symmetric: raises both Strong Drop and Strong Rise |

**Headline finding:** MICE imputation achieves up to 16× reconstruction improvement over mean imputation by exploiting the deterministic algebraic structure of MACD features ($\text{hist} = \text{MACD} - \text{signal}$). SMOTE-generated synthetic samples improve minority-class F1 by nearly 4× over the unhandled baseline for the most extreme class.

### Entry Points

| File | Purpose |
|---|---|
| [`notebooks/01_EDA_and_Target_Creation.ipynb`](./Assignment2/notebooks/01_EDA_and_Target_Creation.ipynb) | EDA and 5-class label construction |
| [`notebooks/02_Imputation_Methods.ipynb`](./Assignment2/notebooks/02_Imputation_Methods.ipynb) | 4-method imputation benchmark |
| [`notebooks/03_Imbalanced_Classification_&_Explainability.ipynb`](./Assignment2/notebooks/03_Imbalanced_Classification_%26_Explainability.ipynb) | Classification strategies and SHAP |
| [`utils/`](./Assignment2/utils/) | Modular Python library (data_loader, imputation, imbalance, evaluation) |
| [`data/df_clean.parquet`](./Assignment2/data/) | Ground-truth dataset (no artificial missingness) |
| [`data/df_missing.parquet`](./Assignment2/data/) | Dataset with 5% MCAR injected into 8 columns |
| [`outputs/models/best_pipeline.joblib`](./Assignment2/outputs/models/) | Persisted best sklearn/imblearn pipeline |
| [`outputs/reports/`](./Assignment2/outputs/reports/) | CSV result tables (imputation quality, strategy comparison) |

📖 **[Full Assignment 2 README →](./Assignment2/README.md)**

---

## How to Run

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Assignment 1

```bash
cd Assignment1

# Run all phases sequentially via the entry-point notebook
jupyter notebook start.ipynb

# Or run each phase independently
python scripts/run_phase1.py   # Data engineering
python scripts/run_phase2.py   # STL decomposition
python scripts/run_phase3.py   # SARIMAX baseline
python scripts/run_phase4.py   # GARCH volatility
python scripts/run_phase5.py   # PatchTST transformer
python scripts/run_phase6.py   # Walk-forward backtest

# Visualisations (or open VisualAnalytics.html directly)
jupyter notebook VisualAnalytics.ipynb
```

### Assignment 2

```bash
cd Assignment2

# Run notebooks in order (01 → 02 → 03)
jupyter notebook notebooks/01_EDA_and_Target_Creation.ipynb
jupyter notebook notebooks/02_Imputation_Methods.ipynb
jupyter notebook notebooks/03_Imbalanced_Classification_\&_Explainability.ipynb
```

---

## Core Technical Stack

| Package | Version | Used In |
|---|---|---|
| `pandas` | 2.2.3 | Both assignments |
| `numpy` | 2.4.4 | Both assignments |
| `scikit-learn` | 1.6.0 | Both assignments |
| `statsmodels` | 0.14.5 | Assignment 1 (SARIMAX, ADF, Ljung-Box) |
| `pmdarima` | 2.0.4 | Assignment 1 (auto_arima order selection) |
| `arch` | 7.2.0 | Assignment 1 (GARCH) |
| `torch` | 2.5.1 | Assignment 1 (PatchTST transformer) |
| `imbalanced-learn` | — | Assignment 2 (SMOTE, ADASYN, imblearn.Pipeline) |
| `shap` | — | Assignment 2 (TreeExplainer) |
| `chronos-forecasting` | 2.2.2 | Assignment 1 (Chronos-T5 reference) |
| `plotly` | 6.3.1 | Both assignments (VisualAnalytics) |

See [`requirements.txt`](./requirements.txt) for the full pinned list.

---

## Full Documentation

| Document | Contents |
|---|---|
| [`Project_Memoire.md`](./Project_Memoire.md) | Master 3,200-line technical document — complete mathematical derivations, implementation details, and empirical interpretations for all phases of both assignments |
| [`PRESENTATION.md`](./PRESENTATION.md) | Full oral presentation script — slide-by-slide guide with formulas, exact numbers, and suggested talking points |

---

*Course: Advanced Machine Learning*
