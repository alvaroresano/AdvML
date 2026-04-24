# Project Memoire: Advanced Financial Time Series Forecasting

## Executive Summary & Goals

This project studies multivariate financial time series forecasting using historical market and macroeconomic data from the Kaggle dataset referenced in the repository README. The objective is not only to produce accurate forecasts, but to build a rigorous quantitative pipeline that combines:

- statistically sound preprocessing,
- interpretable classical time series models,
- volatility-aware risk modeling,
- modern deep learning or foundation-model forecasting,
- and realistic financial backtesting under market frictions.

The project is being developed as an academic and production-style workflow. Each phase will document:

- the mathematical rationale,
- implementation choices,
- empirical findings,
- and implications for forecasting and trading evaluation.

## Dataset Context

- Repository dataset reference: `https://www.kaggle.com/datasets/franciscogcc/financial-data`
- Current workspace evidence comes from the notebooks in `Assignment1/`.
- The source dataset `Assignment1/financial_regression.csv` is now available locally and has been used to initialize the first production preprocessing pipeline.

## Initial EDA Findings From Existing Work

Based on `Assignment1/start.ipynb` and `Assignment1/01_EDA.ipynb`, the current state of the data is:

1. The dataset is treated as a mixed-frequency panel with daily market variables and lower-frequency macroeconomic variables. The notebooks explicitly identify GDP as quarterly and CPI / US rates as monthly using observation-gap analysis.
2. The current preprocessing logic trims the sample to start at `2010-04-01` because early GDP values are missing, then forward-fills `GDP`, `CPI`, and `us_rates_%` so that macro variables align with the daily market calendar.
3. The EDA already checks descriptive statistics, change frequencies, price distributions, log-return distributions, outliers, normalized close-price trajectories, volume behavior, macro-event overlays, and cross-asset / macro correlations. This is a strong exploratory baseline, but stationarity diagnostics, decomposition, formal forecasting models, residual diagnostics, and backtesting infrastructure are not yet implemented in the current notebooks.

## Data Transformations: Initial Rationale

### Why log returns instead of simple returns

For a price series \(P_t\), the simple return is

\[
R_t = \frac{P_t - P_{t-1}}{P_{t-1}}
\]

and the log return is

\[
r_t = \log\left(\frac{P_t}{P_{t-1}}\right) = \log(1 + R_t).
\]

Log returns are usually preferred in quantitative modeling because:

- they are additive over time, which simplifies aggregation and modeling,
- they often stabilize variance better than raw prices,
- and many financial models are formulated naturally in continuously compounded returns.

For small returns, log returns and simple returns are numerically close, since \(\log(1+R_t) \approx R_t\).

### Missing-data handling

The current notebooks use forward fill for macroeconomic variables after removing the earliest rows with missing GDP. This is defensible because macroeconomic indicators are published discretely and remain the latest known value until the next release. We will preserve this logic, but document it explicitly as a mixed-frequency alignment assumption rather than a generic imputation trick.

## Formal Roadmap

### Phase 1: Data Engineering & Statistical Preparation

Objectives:

- restore and validate the source dataset,
- formalize a reproducible data loader,
- transform relevant price variables into log returns,
- test stationarity with the Augmented Dickey-Fuller test,
- engineer technical indicators such as RSI, MACD, and Bollinger Bands.

Deliverables:

- a modular preprocessing pipeline,
- a clean modeling dataset,
- a written explanation of stationarity and feature construction.

### Phase 2: Statistical Decomposition

Objectives:

- apply STL decomposition to selected core series,
- separate level, trend, seasonality, and residual components,
- interpret what structure remains after decomposition.

Deliverables:

- decomposition utilities and plots,
- interpretation of whether seasonality is stable, weak, or regime-dependent.

### Phase 3: Classical Forecasting Baselines

Objectives:

- fit `auto_arima` models using `pmdarima` with AIC-based selection,
- convert the selected specification into a SARIMAX workflow,
- validate residuals using Ljung-Box tests and Q-Q plots.

Deliverables:

- a defensible classical benchmark,
- residual diagnostics showing whether the mean process is adequately modeled.

## Phase 3 Implementation Update

Phase 3 has now been implemented in:

- `Assignment1/src/advml_assignment1/phase3_classical_baseline.py`
- `Assignment1/scripts/run_phase3.py`

The generated outputs are stored in `Assignment1/outputs/phase3/`.

### Why ARIMA and SARIMAX are used as a baseline

Before moving to volatility models and deep learning, a serious time-series project should establish whether a simpler linear model already explains most of the predictable structure. ARIMA-class models are valuable because they are:

- interpretable,
- statistically grounded,
- and useful as a benchmark that later, more complex models must beat.

If a deep model performs only marginally better than a disciplined classical baseline, that is a very different result from beating a weak or poorly specified baseline.

### ARIMA intuition

An ARIMA model is built from three pieces:

- **AR(p)**: autoregressive terms, meaning the series depends on its own past values,
- **I(d)**: integration or differencing, used to remove non-stationarity,
- **MA(q)**: moving-average terms, meaning the series depends on past shocks.

In compact notation:

\[
\phi(B)(1-B)^d y_t = c + \theta(B)\varepsilon_t
\]

where \(B\) is the lag operator, \(\phi(B)\) is the autoregressive polynomial, and \(\theta(B)\) is the moving-average polynomial.

For financial returns, differencing is often unnecessary because returns are already much closer to stationary than price levels. That expectation was already supported by the ADF results from Phase 1.

### What SARIMAX adds

SARIMAX extends ARIMA by allowing:

- optional seasonal structure,
- and exogenous regressors.

Its general idea is:

\[
y_t = c + \beta^\top x_t + \text{ARMA dynamics} + \varepsilon_t.
\]

In this project, the exogenous block is important because the target asset may react not only to its own recent dynamics, but also to:

- related market returns,
- FX moves,
- and macroeconomic changes.

### Why the exogenous block is lagged

The current benchmark targets `nasdaq log_return`. To avoid information leakage, the regressors are lagged so that the model only uses information known before the prediction target is realized.

The exogenous design includes:

- lagged `sp500` return,
- lagged `gold` return,
- lagged `oil` return,
- lagged `eur_usd` return,
- lagged `usd_chf` return,
- lagged GDP growth,
- lagged CPI inflation,
- lagged changes in the US policy rate.

This is a much better design than using contemporaneous same-day features to predict the same-day return, because that would overstate forecasting power.

### Why AIC and `auto_arima` are used

AIC, the Akaike Information Criterion, is defined as

\[
AIC = 2k - 2\log L
\]

where \(k\) is the number of estimated parameters and \(L\) is the maximized likelihood.

The idea is to reward goodness of fit while penalizing unnecessary complexity. A model with more parameters can fit the training data better almost by construction, so AIC helps prevent choosing an over-parameterized specification.

In this project, `pmdarima.auto_arima` searches candidate ARIMA orders and selects the one with the best AIC. The chosen order is then refit using `statsmodels` SARIMAX so that full diagnostics and forecasting outputs can be produced cleanly.

### Why the model is non-seasonal

STL decomposition in Phase 2 showed very weak weekly seasonal structure for most assets. Therefore, the classical baseline uses a non-seasonal SARIMAX specification as the default benchmark. This is a principled choice:

- if the decomposition does not show strong stable seasonality,
- and if a simpler non-seasonal model explains the mean process adequately,
- then adding seasonal terms by default would be statistically unmotivated complexity.

### Residual diagnostics: why they matter

Fitting a model is not enough. The residuals must also be checked.

#### Ljung-Box test

The Ljung-Box test evaluates whether residual autocorrelation remains across multiple lags. The null hypothesis is that the residuals are independently distributed with no autocorrelation up to the tested lag.

If the p-value is small, then the model has failed to capture all the linear dependence structure.

#### Q-Q plot

A Q-Q plot compares the empirical quantiles of the residuals to the quantiles of a normal distribution. If the residuals are approximately normal, the points lie near a straight line. Large departures in the tails indicate heavy-tailed behavior or skewness.

In finance, tail departures are common because market returns often exhibit crash risk, asymmetry, and outlier behavior.

### Empirical findings from Phase 3

The benchmark was run on:

- target: `nasdaq log_return`
- training sample: 3351 observations
- holdout test sample: 252 observations
- test period: from `2023-10-17` to `2024-10-18`

`auto_arima` selected:

- order: `(0, 0, 0)`
- seasonal order: `(0, 0, 0, 0)`

This is a meaningful outcome. It means that once the return transformation and lagged exogenous block are in place, the data does not support adding extra AR or MA terms to the mean equation under AIC.

Test-set performance:

- RMSE: approximately `0.01103`
- MAE: approximately `0.00824`
- directional accuracy: approximately `0.5278`

Coefficient interpretation:

- the lagged `sp500` return is the strongest statistically significant regressor in the fitted model,
- the intercept is small but statistically significant,
- most other exogenous coefficients are not strongly significant in this linear benchmark.

Residual diagnostics:

- Ljung-Box at lag 5 is borderline (`p ≈ 0.0588`),
- Ljung-Box at lags 10 and 20 strongly rejects no residual autocorrelation,
- Jarque-Bera strongly rejects normality,
- and the Q-Q plot shows clear tail departures from Gaussian behavior.

### Interpretation of the Phase 3 result

The key lesson is not that the model failed, but that it reached a meaningful limit:

- the mean process appears only weakly predictable with simple linear ARMA dynamics,
- the exogenous block explains some structure,
- but substantial residual dependence and heavy-tailed behavior remain.

This is exactly the type of result that motivates the next project stages:

- GARCH for conditional variance and volatility clustering,
- and richer forecasting models for nonlinear or cross-series structure.

## Phase 2 Implementation Update

Phase 2 has now been implemented with STL decomposition code in:

- `Assignment1/src/advml_assignment1/phase2_stl_decomposition.py`
- `Assignment1/scripts/run_phase2.py`

The outputs are stored under `Assignment1/outputs/phase2/` and include:

- a long-form components table,
- an asset-level summary table,
- and one decomposition plot per asset.

### Why STL is useful here

Financial prices often mix together multiple types of structure:

- a low-frequency trend,
- a possible repeating seasonal pattern,
- and irregular shocks.

Looking only at the raw series makes it hard to tell which behavior belongs to which source. STL gives a way to separate these layers so that later forecasting models can be designed more intelligently.

### Why we decompose log prices instead of raw prices

If a price process behaves approximately multiplicatively, a simple schematic form is

\[
P_t \approx T_t \times S_t \times E_t
\]

where \(T_t\) is trend, \(S_t\) is a seasonal factor, and \(E_t\) is an irregular component. Taking logs converts this into an additive representation:

\[
\log P_t \approx \log T_t + \log S_t + \log E_t.
\]

STL is an additive decomposition method, so applying it to log prices is much more coherent than applying it directly to level prices whose variability scales with their magnitude.

### STL mechanics

STL means Seasonal-Trend decomposition using Loess. It models a series as

\[
y_t = T_t + S_t + R_t
\]

where:

- \(T_t\) is the smooth trend,
- \(S_t\) is the repeating seasonal component,
- \(R_t\) is the remainder or residual.

The key idea is that Loess smoothing fits local regressions around each time point. Instead of assuming one global polynomial or one rigid sinusoidal seasonal rule, STL lets the local structure adapt over time.

### Choice of period

For this project, the decomposition uses:

- the cleaned trading-day dataset from Phase 1,
- log-close prices as the observed series,
- `period = 5`, which corresponds to one trading week,
- and `robust = True`, so that large outliers have less influence.

This choice is economically sensible because:

- weekends and market holidays were removed in Phase 1, so the index is a trading-time calendar,
- five observations therefore represent a natural weekly cycle in trading time,
- and robust fitting is important because financial markets contain large shocks that can otherwise distort the seasonal estimate.

### Strength metrics used for interpretation

To interpret the decomposition quantitatively, the project computes:

\[
F_T = \max\left(0, 1 - \frac{\operatorname{Var}(R_t)}{\operatorname{Var}(T_t + R_t)}\right)
\]

for trend strength and

\[
F_S = \max\left(0, 1 - \frac{\operatorname{Var}(R_t)}{\operatorname{Var}(S_t + R_t)}\right)
\]

for seasonal strength.

These scores are close to 1 when the corresponding component explains much more than the residual noise, and close to 0 when it does not.

### Empirical findings from Phase 2

The STL decomposition was run on all seven close-price assets:

- `sp500`
- `nasdaq`
- `silver`
- `oil`
- `platinum`
- `palladium`
- `gold`

Main findings:

- Average trend strength across assets is approximately `0.9977`, which is extremely high.
- Average seasonal strength is only about `0.0563`, which is very weak overall.
- For six of the seven assets, seasonal strength is effectively zero under a weekly trading-calendar decomposition.
- Oil is the only asset with a materially positive seasonal-strength score (`0.3941`), but inspection shows this is driven by a few extreme spikes rather than a stable weekday pattern.

This means the decomposition is telling a clear story:

- the dominant structure in these asset prices is trend,
- the residual component captures shocks and idiosyncratic variation,
- and there is little evidence of a strong, persistent weekly seasonal pattern for most assets.

This is an important result for the next phases. It suggests that:

- classical forecasting models should focus much more on trend, autocorrelation, and shock structure than on strong deterministic seasonality,
- and any seasonal terms added later should be justified empirically rather than assumed by default.

## Phase 1 Implementation Update

Phase 1 has now been implemented in modular Python code. The first production components are:

- `Assignment1/src/advml_assignment1/phase1_data_engineering.py`
- `Assignment1/scripts/run_phase1.py`

The pipeline performs the following steps:

1. Load the raw CSV, sort by date, and validate the required schema.
2. Trim the dataset to begin at `2010-04-01`, consistent with the original EDA decision that avoids the early missing-GDP segment.
3. Remove non-trading rows, defined as rows where all close-price columns are missing.
4. Forward-fill macroeconomic variables (`GDP`, `CPI`, `us_rates_%`) to align lower-frequency releases with the daily trading calendar.
5. Engineer technical features from each asset close series:
   - log returns,
   - RSI(14),
   - MACD(12, 26, 9),
   - Bollinger Bands(20, 2 standard deviations).
6. Run ADF tests on both price levels and log-return series.
7. Export cleaned, featured, and complete-case modeling datasets together with an ADF summary table.

### Empirical results from the Phase 1 run

The live pipeline run produced the following dataset sizes:

- Raw observations: 3904 rows.
- Rows removed before `2010-04-01`: 55.
- Non-trading rows removed after trimming: 183.
- Cleaned trading observations: 3666.
- Complete-case modeling observations after technical-indicator warm-up and remaining daily FX gaps: 3605.

### Stationarity findings

ADF results are consistent with standard financial theory:

- For all seven close-price series (`sp500`, `nasdaq`, `silver`, `oil`, `platinum`, `palladium`, `gold`), the price-level series fail to reject the unit-root null at the 5% level.
- For all seven log-return series, the ADF test strongly rejects the unit-root null at the 5% level.

This is an important modeling conclusion: the raw price processes behave as non-stationary series, while log returns are much better candidates for mean-modeling and volatility-modeling.

## Methodology Notes

### Augmented Dickey-Fuller (ADF)

The ADF test checks whether a time series contains a unit root. A simplified regression form is

\[
\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^{p}\phi_i \Delta y_{t-i} + \varepsilon_t.
\]

The null hypothesis is that the series has a unit root, which implies non-stationarity. Rejecting the null supports stationarity. In finance, raw prices are usually non-stationary, while returns are often closer to stationary.

### STL Decomposition

STL decomposes a series into:

\[
y_t = T_t + S_t + R_t
\]

where \(T_t\) is trend, \(S_t\) is seasonal structure, and \(R_t\) is residual noise. This helps isolate smoother long-run movement from recurring seasonal patterns and idiosyncratic shocks.

In this project, STL is applied to log prices on the trading-day calendar with a period of 5 observations. Empirically, the decomposition shows very strong trend structure and very weak seasonal structure for most assets.

### ARMA / ARIMA / SARIMAX

An ARMA model combines autoregressive and moving-average dynamics:

\[
y_t = \sum_{i=1}^{p}\phi_i y_{t-i} + \varepsilon_t + \sum_{j=1}^{q}\theta_j \varepsilon_{t-j}.
\]

ARIMA extends this by differencing to handle non-stationarity. SARIMAX further allows seasonal structure and exogenous regressors, which is appropriate for financial settings where macro variables may help explain returns or transformed prices.

In the implemented benchmark, AIC-driven `auto_arima` selected a non-seasonal `(0,0,0)` mean specification once lagged exogenous variables were included, which suggests very limited additional ARMA structure in the target return series.

### GARCH(1,1)

After modeling the conditional mean, the conditional variance can be modeled as

\[
\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2.
\]

This captures volatility clustering, a core stylized fact of financial returns. We will fit GARCH on the ARIMA residuals to separate mean forecasting from risk forecasting.

### Deep Learning / Foundation Forecasting

The project will compare classical statistical baselines with a modern sequence model, likely PatchTST or Chronos-style forecasting. The purpose is not to assume deep models are automatically better, but to test whether they extract nonlinear cross-series structure beyond what classical methods capture.

### Backtesting and Financial KPIs

Forecasting performance will be evaluated not only with error metrics, but also with finance-specific metrics:

- Hit Rate: directional accuracy of predicted returns,
- Sharpe Ratio: return per unit of volatility,
- Maximum Drawdown: worst cumulative loss from a peak,
- and transaction-cost-aware PnL under slippage and commissions.

### Technical Indicators Used in Phase 1

#### Relative Strength Index (RSI)

RSI is a bounded momentum oscillator. If \(\Delta P_t = P_t - P_{t-1}\), then gains and losses are separated as

\[
G_t = \max(\Delta P_t, 0), \qquad L_t = \max(-\Delta P_t, 0).
\]

Using Wilder-style smoothed averages over a window \(n\),

\[
RS_t = \frac{\text{AvgGain}_t}{\text{AvgLoss}_t}
\]

and

\[
RSI_t = 100 - \frac{100}{1 + RS_t}.
\]

RSI helps detect overbought and oversold conditions, but in this project it is treated as a quantitative feature rather than a standalone trading rule.

#### MACD

MACD measures trend and momentum by comparing fast and slow exponential moving averages:

\[
MACD_t = EMA_{12}(P_t) - EMA_{26}(P_t).
\]

The signal line is a smoothed version of MACD:

\[
Signal_t = EMA_{9}(MACD_t),
\]

and the histogram is

\[
Hist_t = MACD_t - Signal_t.
\]

These quantities help quantify whether price momentum is accelerating or decelerating.

#### Bollinger Bands

For a rolling window \(n\), Bollinger Bands are defined as

\[
Middle_t = \frac{1}{n}\sum_{i=0}^{n-1} P_{t-i},
\]

\[
Upper_t = Middle_t + k \sigma_t,
\]

\[
Lower_t = Middle_t - k \sigma_t,
\]

where \(\sigma_t\) is the rolling standard deviation and \(k\) is usually set to 2. The bands provide a normalized way to measure how far a price has moved relative to its recent local volatility.

## Code Architecture: Initial Design

The implementation will move from notebook-only analysis to a modular pipeline with clear responsibilities:

- data loading and validation,
- preprocessing and feature engineering,
- decomposition and diagnostics,
- baseline statistical models,
- volatility models,
- deep forecasting models,
- and backtesting / evaluation.

Each module will expose reusable classes or functions so that experiments remain reproducible and easy to compare.

## Current Project State

- `Assignment1/start.ipynb` and `Assignment1/01_EDA.ipynb` were reviewed.
- Phase 1 preprocessing and stationarity code has been implemented and validated on the live dataset.
- Phase 2 STL decomposition has been implemented, exported, and interpreted on the live dataset.
- Phase 3 classical benchmarking has been implemented with `pmdarima` order selection, a SARIMAX fit, residual diagnostics, and updated visual analytics.
- Phase 4 will add volatility modeling through GARCH on the classical-model residuals.

## Bibliography / Sources

1. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., and Ljung, G. M. *Time Series Analysis: Forecasting and Control*.
2. Hyndman, R. J., and Athanasopoulos, G. *Forecasting: Principles and Practice*.
3. Hamilton, J. D. *Time Series Analysis*.
4. Dickey, D. A., and Fuller, W. A. (1979). Distribution of the estimators for autoregressive time series with a unit root.
5. Cleveland, R. B., Cleveland, W. S., McRae, J. E., and Terpenning, I. (1990). STL: A seasonal-trend decomposition procedure based on loess.
6. Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation.
7. Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity.
8. Wilder, J. W. *New Concepts in Technical Trading Systems*.
9. Bollinger, J. *Bollinger on Bollinger Bands*.
10. Appel, G. *Technical Analysis: Power Tools for Active Investors*.
11. Repository README dataset reference: Kaggle financial data dataset by `franciscogcc`.
