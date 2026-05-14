---
name: project-context
description: Advanced Financial Time Series Forecasting project — full pipeline structure, implemented phases, key empirical results, and notebook inventory
metadata:
  type: project
---

Advanced ML class project: multivariate financial time series forecaster on Kaggle dataset (SP500, NASDAQ, Gold, Silver, Oil, Platinum, Palladium + macro/FX).

**Pipeline (Assignment1/):**
- Phase 1: Data engineering, ADF tests, technical features (RSI, MACD, Bollinger) → `outputs/phase1/`
- Phase 2: STL decomposition on log-close prices, period=5 → strong trend (~0.9977), very weak seasonality (~0.056) → `outputs/phase2/`
- Phase 3: SARIMAX (0,0,0) with 8 lagged exogenous, target=nasdaq log_return. RMSE=0.01103, hit=52.8% → `outputs/phase3/`
- Phase 4: GARCH(1,1) Student-t on Phase 3 residuals. persistence=0.9916, half-life=82.4 periods → `outputs/phase4/`
- Phase 5: PatchTST-style transformer (60-day lookback, 10-day patches, 33 features). RMSE=0.01090, hit=57.1%, best epoch=9 → `outputs/phase5/`
- Phase 6: 5-fold rolling backtest. SARIMAX: Sharpe 0.525, net +65.8%. PatchTST: Sharpe 0.496, net +59.5%. Key finding: better stats ≠ better economics → `outputs/phase6/`

**Standalone notebooks:**
- `01_EDA.ipynb`: Mixed-frequency detection, quality checks, distributions, correlations
- `03_Forecasting_LSTM_&_Chronos.ipynb`: Hybrid STL-LSTM targeting SP500 residuals (NOT NASDAQ). Chronos-T5-base zero-shot (5-step). 3-way volatility comparison (GARCH vs LSTM proxy vs Chronos width). Only 5-day evaluation window — statistically fragile.
- `VisualAnalytics.ipynb`: 8-block interactive Plotly companion covering all phases

**Key docs:**
- `Project_Memoire.md`: Master documentation — 2644 lines, all phases + LSTM-Chronos + VisualAnalytics sections added

**Why:** LSTM targets SP500 residuals (not NASDAQ) — artifact of first available STL asset, not intentional design choice. This breaks direct metric comparison with other phases.
