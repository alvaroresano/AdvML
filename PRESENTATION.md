# Advanced Financial Time Series Forecasting
## Full Technical Presentation Guide

This document is your complete oral presentation script. Each section maps to one slide or speaking block. For every block you will find: the core message, the mathematics, the exact empirical numbers, what to say out loud, and which visual to show.

---

## SLIDE 1 — Title and Project Goal

**Core message:** We built a complete quantitative pipeline that forecasts NASDAQ daily log returns, progressing from classical statistics through deep learning, and evaluated the result under realistic trading conditions.

**What to say:**

> "This project studies multivariate financial time series forecasting using a dataset of seven assets, two FX pairs, and three macroeconomic variables from 2010 to 2024. The goal was not just to produce a prediction number — it was to build a rigorous pipeline that is statistically defensible at every stage. That means starting with data engineering, moving through stationarity testing, decomposition, classical benchmarks, volatility modeling, and deep learning, and ending with a financial evaluation under actual transaction costs. Every modeling choice is justified by theory, and every result is critically interpreted."

**Key claims to anchor your opening:**

- Dataset: 14 years of daily trading data (3,605 modeling observations after cleaning)
- Target variable: next-day NASDAQ log return
- Models implemented: SARIMAX, GARCH(1,1), PatchTST-style transformer, Hybrid LSTM, Amazon Chronos
- Final evaluation: 5-fold rolling walk-forward backtest with 5 basis points of transaction costs

---

## SLIDE 2 — Pipeline Architecture

**Core message:** The project is organized as a disciplined six-phase pipeline where each phase builds on the output of the previous one.

```
Raw CSV
  ↓
Phase 1: Data Engineering + ADF Tests
  ↓
Phase 2: STL Decomposition
  ↓
Phase 3: SARIMAX Classical Baseline (mean forecasting)
  ↓
Phase 4: GARCH(1,1) Volatility Modeling (variance forecasting)
  ↓
Phase 5: PatchTST-style Transformer (nonlinear mean forecasting)
  ↓
Phase 6: Rolling Walk-Forward Backtest (economic evaluation)
```

**What to say:**

> "The pipeline is sequential and intentional. Each phase answers a specific question. Phase 1 asks: is the data usable and stationary? Phase 2 asks: what structure exists in the series? Phase 3 asks: can a classical linear model explain the mean? Phase 4 asks: can we model the changing risk around that mean? Phase 5 asks: does a modern sequence model beat the linear benchmark? Phase 6 asks: do any of these improvements survive repeated historical evaluation under realistic costs?
>
> This progression is important because it forces each complex model to justify its added complexity over a disciplined simpler baseline. A model that only beats a weak baseline proves very little."

---

## SLIDE 3 — The Dataset

**Core message:** The dataset is a mixed-frequency panel — daily market data and lower-frequency macro releases — which requires careful alignment before any modeling can begin.

**Assets in the dataset:**

| Category | Variables |
| --- | --- |
| Equity indices | SP500, NASDAQ (OHLCV) |
| Commodities | Gold, Silver, Oil, Platinum, Palladium (OHLCV) |
| FX rates | EUR/USD, USD/CHF (daily close) |
| Macro (quarterly) | GDP |
| Macro (monthly) | CPI, US federal funds rate |

**Time period:** 2010-04-01 to 2024-10-18

**Raw dataset:** 3,904 rows × 47 columns

**What to say:**

> "The dataset is sourced from Kaggle and contains 47 columns covering price, volume, FX, and macroeconomic data. The critical complication is that these variables update at different frequencies. OHLCV data changes every trading day. The CPI and federal funds rate are published monthly. GDP is published quarterly. If we treat all of these as if they had the same frequency, we are making a fundamental error.
>
> The EDA notebook — and this is an important step — explicitly confirms these frequencies by computing the gap between consecutive non-null observations. GDP shows a mean gap of 91 days. CPI and rates show a mean gap of about 30 days. This is the empirical basis for our mixed-frequency handling strategy."

---

## SLIDE 4 — Phase 1: Data Engineering and Cleaning

**Core message:** Three principled decisions were made to produce a clean, analysis-ready dataset, each justified by the specific properties of the data.

**Decision 1 — Trim to 2010-04-01:**
GDP has its first non-null observation on 2010-04-01. Rows before this date have a missing macro variable that cannot be meaningfully forward-filled from an earlier known value because there is no earlier known value. Trimming removes 55 rows.

**Decision 2 — Remove non-trading rows:**
Rows where every close-price column is NaN are market holidays or weekends that entered the dataset. These 183 rows are dropped because they contain no market information. The result is a pure trading-calendar dataset with 3,666 observations.

**Decision 3 — Forward-fill macro variables:**
After the first quarterly GDP release on 2010-04-01, the value is held constant until the next release arrives. This is not a generic imputation trick — it is the correct economic behavior. When the Fed sets the funds rate at 5.25%, that rate is 5.25% every day until the FOMC next meets. A macroeconomic release holds until it is superseded.

**What to say:**

> "After cleaning we have 3,666 trading-day observations. The final modeling dataset has 3,605 rows — the difference comes from the warm-up period needed for rolling technical indicators like the 26-day MACD slow EMA and the 20-day Bollinger window. We drop these warm-up rows only at the end, after all features have been computed on the full history, so no signal from the past is lost."

---

## SLIDE 5 — Phase 1: Why Log Returns Instead of Raw Prices

**Core message:** Raw prices are non-stationary and multiplicative. Log returns solve both problems.

**The mathematics:**

For a price series $P_t$, the simple return is:

$$R_t = \frac{P_t - P_{t-1}}{P_{t-1}}$$

The log return is:

$$r_t = \log\!\left(\frac{P_t}{P_{t-1}}\right) = \log(1 + R_t)$$

**Why log returns are preferred:**

1. **Additivity over time.** The cumulative log return over $n$ periods is the sum of daily log returns: $r_{1 \to n} = \sum_{t=1}^n r_t$. Simple returns do not add — they multiply. This makes log returns trivial to aggregate.

2. **Symmetry.** A 50% drop followed by a 100% gain returns the price to its origin only in simple-return arithmetic. In log-return arithmetic, a drop of $-\log 2$ followed by a gain of $+\log 2$ correctly sums to zero.

3. **Multiplicative-to-additive decomposition.** Prices follow approximately geometric Brownian motion: $P_t = P_{t-1} \cdot e^{r_t}$. Taking logs converts this multiplicative process into an additive one, which is what ARIMA, GARCH, and regression-based models assume.

4. **STL compatibility.** STL decomposition requires an additive structure: $y_t = T_t + S_t + R_t$. Applying STL to log prices achieves this because $\log P_t \approx \log T_t + \log S_t + \log E_t$ when prices behave multiplicatively.

**What to say:**

> "For small returns, log returns and simple returns are numerically close — they differ by approximately $R_t^2/2$, which is negligible for daily moves of less than 2–3%. But their mathematical properties diverge sharply over longer horizons and when used inside statistical models. For this project, using log returns is not a convention — it is the theoretically correct transformation for the models we apply."

---

## SLIDE 6 — Phase 1: Technical Indicators

**Core message:** Three families of technical indicators are engineered from close prices. They serve as structured input features for the deep learning model.

**RSI(14) — Relative Strength Index:**

$$RSI_t = 100 - \frac{100}{1 + RS_t}, \quad RS_t = \frac{\overline{G}_{14}}{\overline{L}_{14}}$$

where $\overline{G}_{14}$ and $\overline{L}_{14}$ are Wilder-smoothed 14-period average gains and losses. RSI lives on [0, 100]. Values above 70 suggest overbought conditions; below 30, oversold. In this project it is a quantitative signal, not a trading rule.

**MACD(12, 26, 9) — Moving Average Convergence/Divergence:**

$$MACD_t = EMA_{12}(P_t) - EMA_{26}(P_t)$$
$$Signal_t = EMA_9(MACD_t)$$
$$Histogram_t = MACD_t - Signal_t$$

The histogram quantifies whether short-term momentum is accelerating (positive and growing) or decelerating (positive and shrinking or negative).

**Bollinger z-score (window 20, 2σ):**

$$z_t = \frac{P_t - \mu_t}{\sigma_t}$$

where $\mu_t$ and $\sigma_t$ are the rolling 20-day mean and standard deviation. This is not the standard "band" representation — we use the z-score directly because it is a scale-free normalized distance that works as a feature without additional preprocessing.

**What to say:**

> "These three indicators capture different aspects of market dynamics. RSI captures momentum pressure on a bounded scale. MACD captures the relative strength of short- versus medium-term trends. The Bollinger z-score captures how far prices have moved from their recent local average in units of recent volatility. Together they give the model information about overbought/oversold conditions, trend direction, and mean-reversion pressure simultaneously — all from a single price series."

---

## SLIDE 7 — Phase 1: ADF Stationarity Test

**Core message:** Statistical theory requires stationarity for ARIMA-class models. The ADF test formally confirms that log returns are stationary and raw prices are not.

**The null and alternative hypotheses:**

$$H_0: \text{unit root present} \Rightarrow \text{non-stationary}$$
$$H_1: \text{no unit root} \Rightarrow \text{stationary}$$

Decision rule: reject $H_0$ when $p < 0.05$.

The ADF test fits an augmented regression of the form:

$$\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^{p}\phi_i \Delta y_{t-i} + \varepsilon_t$$

The test statistic for $\gamma = 0$ is negative; more negative means stronger rejection of the unit root.

**Empirical results (all 7 assets):**

| Asset | Price level ADF stat | Price p-value | Log return ADF stat | Return p-value | Stationary? |
| --- | --- | --- | --- | --- | --- |
| NASDAQ | +1.44 | 0.997 | -13.62 | 1.79 × 10⁻²⁵ | Returns: Yes |
| SP500 | +1.20 | 0.996 | -13.13 | 1.50 × 10⁻²⁴ | Returns: Yes |
| Gold | +0.34 | 0.979 | -61.00 | ~0 | Returns: Yes |
| Silver | -1.85 | 0.357 | -59.98 | ~0 | Returns: Yes |
| Oil | -0.28 | 0.929 | -11.24 | 1.82 × 10⁻²⁰ | Returns: Yes |
| Platinum | -1.94 | 0.315 | -14.38 | 9.03 × 10⁻²⁷ | Returns: Yes |
| Palladium | -1.58 | 0.493 | -11.64 | 2.21 × 10⁻²¹ | Returns: Yes |

**What to say:**

> "The results are unambiguous. Every single price-level series fails to reject the unit-root null — p-values range from 0.31 to 0.997, all far above 0.05. Every single log-return series strongly rejects the unit-root null — p-values are between 10⁻²⁰ and essentially zero. Some of these test statistics, like gold at -61 and silver at -60, are extraordinary — they indicate returns that behave almost like pure white noise with no persistent trend component at all.
>
> One clarification that is important for this chart: the bar chart in the VisualAnalytics notebook does NOT plot raw p-values. It plots $-\log_{10}(p)$. This transformation maps small p-values to tall bars and large p-values to short bars, so 'taller bar = stronger evidence against unit root' is correct — but only because of the log transformation. The threshold line sits at $-\log_{10}(0.05) \approx 1.301$."

**Visual:** Open `VisualAnalytics.ipynb`, Block 2 ADF bar chart.

---

## SLIDE 8 — Phase 2: STL Decomposition

**Core message:** STL separates each log-price series into trend, seasonal, and residual components. The key finding is that these assets are driven by trend and shocks, not by stable weekly seasonality.

**Why decompose log prices, not raw prices:**

If a price process is multiplicative:
$$P_t \approx T_t \cdot S_t \cdot E_t$$

then taking logs gives an additive representation:
$$\log P_t \approx \log T_t + \log S_t + \log E_t$$

STL requires additive structure, so it is applied to log prices.

**STL mechanics:**

STL (Seasonal-Trend decomposition using Loess) fits local polynomial regressions iteratively:

$$y_t = T_t + S_t + R_t$$

where $T_t$ is the smooth trend, $S_t$ is the repeating seasonal component, and $R_t$ is the residual. Key parameters used:

- Period = 5 (one trading week)
- Robust = True (large outliers have reduced influence on the seasonal estimate)

**Strength metrics:**

$$F_T = \max\!\left(0,\; 1 - \frac{\text{Var}(R_t)}{\text{Var}(T_t + R_t)}\right), \quad F_S = \max\!\left(0,\; 1 - \frac{\text{Var}(R_t)}{\text{Var}(S_t + R_t)}\right)$$

Close to 1: that component dominates. Close to 0: it is negligible.

---

## SLIDE 9 — Phase 2: STL Empirical Results

**Empirical results:**

| Asset | Trend strength | Seasonal strength | Residual std | Seasonal amplitude |
| --- | --- | --- | --- | --- |
| NASDAQ | **0.9996** | 0.0000 | 0.0137 | 0.069 |
| SP500 | **0.9993** | 0.0000 | 0.0120 | 0.115 |
| Gold | **0.9973** | 0.0000 | 0.0101 | 0.077 |
| Silver | **0.9947** | 0.0000 | 0.0198 | 0.080 |
| Platinum | **0.9965** | 0.0000 | 0.0159 | 0.079 |
| Palladium | **0.9976** | 0.0000 | 0.0230 | 0.086 |
| Oil | **0.9986** | **0.3941** | 0.0263 | **1.714** |

**What to say:**

> "The results are decisive. All seven assets show trend strength above 0.994 — the trend component dominates. Seasonal strength is essentially zero for six of seven assets under the 5-day trading-week cycle. This means there is no reliable 'Monday effect' or 'Friday effect' in these series that STL can detect as a stable repeating pattern.
>
> Oil is the only outlier with a seasonal strength of 0.394, but look at the seasonal amplitude: 1.714. That is an enormous amplitude compared to the residual standard deviation of 0.026. This is not a real weekly calendar pattern — it is a few extreme episodes, most notably the April 2020 negative-price event, that are contaminating the seasonal estimate. Visually, the oil seasonal component does not show a clean repeating weekday shape.
>
> The modeling implication is direct: we do not add seasonal terms to the SARIMAX model by default. That would be statistically unmotivated complexity."

**Visual:** Open `VisualAnalytics.ipynb`, Block 4, NASDAQ STL dashboard and the trend/seasonal strength bar chart.

---

## SLIDE 10 — Phase 3: SARIMAX Classical Baseline — Theory

**Core message:** SARIMAX is the correct and complete name for our classical benchmark. `auto_arima` is only the order-selection tool; SARIMAX is the fitted model.

**ARIMA theory:**

$$\phi(B)(1-B)^d y_t = c + \theta(B)\varepsilon_t$$

- $AR(p)$: $y_t$ depends on its own $p$ past values
- $I(d)$: $d$ differences needed to remove non-stationarity
- $MA(q)$: $y_t$ depends on $q$ past shock terms

**SARIMAX extension:**

$$y_t = c + \boldsymbol{\beta}^\top \mathbf{x}_t + \text{ARMA dynamics} + \varepsilon_t$$

It adds two things: (1) optional seasonal structure at period $m$, and (2) exogenous regressors $\mathbf{x}_t$.

**AIC-based order selection:**

$$AIC = 2k - 2\log\hat{L}$$

where $k$ is the number of estimated parameters and $\hat{L}$ is the maximized log-likelihood. `pmdarima.auto_arima` searches all candidate ARIMA orders up to $(p_{\max}, d_{\max}, q_{\max}) = (5, 1, 5)$ and selects the specification with the smallest AIC. The chosen order is then refit as a `statsmodels.SARIMAX` model to obtain full diagnostics.

**What to say:**

> "A critical distinction: when someone asks 'are you using ARIMA or SARIMAX?', the correct answer is SARIMAX, because our model includes exogenous regressors. `auto_arima` is a selection procedure, not the benchmark model itself. The benchmark is a SARIMAX whose ARIMA order was chosen by AIC minimization. AIC is the right criterion here because it penalizes unnecessary parameters — it will reject an AR(2) if the second lag adds less fit than it costs in complexity."

---

## SLIDE 11 — Phase 3: The Exogenous Block

**Core message:** All 8 exogenous regressors are lagged by exactly one trading day. This prevents information leakage — the model only uses what was known before the target return occurred.

**The design matrix:**

| Column | Transformation | Source |
| --- | --- | --- |
| `sp500_ret_l1` | `log(SP500_t / SP500_{t-1}).shift(1)` | Phase 1 modeling data |
| `gold_ret_l1` | `log(Gold_t / Gold_{t-1}).shift(1)` | Phase 1 modeling data |
| `oil_ret_l1` | `log(Oil_t / Oil_{t-1}).shift(1)` | Phase 1 modeling data |
| `eur_usd_ret_l1` | `log(EURUSD_t / EURUSD_{t-1}).shift(1)` | Computed inline (raw FX) |
| `usd_chf_ret_l1` | `log(USDCHF_t / USDCHF_{t-1}).shift(1)` | Computed inline (raw FX) |
| `gdp_growth_l1` | `log(GDP_t / GDP_{t-1}).shift(1)` | Phase 1 (forward-filled) |
| `cpi_inflation_l1` | `log(CPI_t / CPI_{t-1}).shift(1)` | Phase 1 (forward-filled) |
| `rate_change_l1` | `(rate_t - rate_{t-1}).shift(1)` | Simple diff (not log) |

**Why the rate uses a simple diff instead of log return:**

The federal funds rate can be zero (e.g., 0.07% in 2011). `log(0.07 / 0.07)` is fine, but `log(0.20 / 0.07)` when rates are near zero produces enormous values that are not interpretable as a proportional economic change. A simple first difference — from 0.07% to 0.20% is +0.13 percentage points — is both numerically stable and directly economically interpretable as a rate move.

**All features are standardized using training-sample statistics before fitting:**

$$\tilde{x}_{it} = \frac{x_{it} - \bar{x}_i^{train}}{s_i^{train}}$$

This puts all predictors on the same numerical scale for estimation. It does not change the model's economic interpretation but stabilizes the optimization. Coefficients are then read as "effect of a one-training-sample-standard-deviation move."

**What to say:**

> "The `.shift(1)` call is the single most important line in the exogenous block. Without it, the model uses same-day SP500 returns to predict same-day NASDAQ returns. These are contemporaneously correlated at about 0.95, so the model would appear to forecast perfectly in training — but in production, the SP500 close is not known until after the NASDAQ closes. The shift enforces the realistic information constraint."

---

## SLIDE 12 — Phase 3: SARIMAX Results

**Core message:** `auto_arima` selects order (0, 0, 0). This is not a trivial result — it means the lagged exogenous block already explains all the mean structure that AIC can justify adding ARMA terms for.

**Selected specification:**

- ARIMA order: **(0, 0, 0)** — no autoregressive or moving-average terms
- Seasonal order: **(0, 0, 0, 0)** — consistent with Phase 2 finding of no seasonality
- Constant term: included (`trend='c'`)
- Model family: SARIMAX (because exogenous regressors are present)

This is equivalent to a linear regression on lagged exogenous variables plus an intercept.

**Test-set performance (252 trading days, 2023-10-17 to 2024-10-18):**

| Metric | Value |
| --- | --- |
| RMSE | **0.01103** |
| MAE | **0.00824** |
| Directional accuracy | **52.78%** |
| Training observations | 3,351 |

**Coefficient estimates (standardized predictors):**

| Predictor | Coeff | Std error | z-score | p-value | Significant? |
| --- | --- | --- | --- | --- | --- |
| Intercept | +0.000583 | 0.000235 | +2.48 | 0.013 | **Yes** |
| `sp500_ret_l1` | −0.001525 | 0.000133 | −11.47 | 1.9 × 10⁻³⁰ | **Yes** |
| `gold_ret_l1` | +0.000354 | 0.000185 | +1.91 | 0.056 | Borderline |
| `oil_ret_l1` | −0.000028 | 0.000294 | −0.10 | 0.924 | No |
| `eur_usd_ret_l1` | +0.000345 | 0.000254 | +1.36 | 0.175 | No |
| `usd_chf_ret_l1` | +0.000328 | 0.000246 | +1.33 | 0.183 | No |
| `gdp_growth_l1` | +0.000067 | 0.000327 | +0.21 | 0.837 | No |
| `cpi_inflation_l1` | +0.000221 | 0.000259 | +0.85 | 0.393 | No |
| `rate_change_l1` | −0.000280 | 0.000201 | −1.40 | 0.162 | No |

**What to say:**

> "The only strongly significant predictor is the lagged SP500 return, with a z-score of -11.47. Everything else is statistically weak. But this needs to be interpreted carefully. The negative coefficient does NOT mean NASDAQ and SP500 move in opposite directions — they are contemporaneously correlated at about 0.95. The negative sign on the lagged SP500 coefficient, after controlling for all other variables, suggests a modest conditional reversal effect in the linear mean equation. This is a model-based conditional relationship, not a universal market law.
>
> The (0,0,0) order is meaningful. It tells us that once you apply the return transformation and add the lagged exogenous block, there is no additional ARMA structure in the mean that AIC justifies adding. The predictable structure in the conditional mean is limited to what the cross-asset lag structure captures."

**Visual:** Open `VisualAnalytics.ipynb`, Block 5, forecast timeline and coefficient chart.

---

## SLIDE 13 — Phase 3: Residual Diagnostics

**Core message:** The mean model is not adequate — residuals show autocorrelation at longer lags and strongly non-Gaussian tails. This motivates Phase 4.

**Ljung-Box test** checks whether residual autocorrelation exists up to lag $k$:

$$LB_k = n(n+2)\sum_{j=1}^{k}\frac{\hat{\rho}_j^2}{n-j}$$

Under $H_0$ (no autocorrelation), $LB_k \sim \chi^2(k)$.

**Results:**

| Lag | LB statistic | p-value | Conclusion |
| --- | --- | --- | --- |
| 5 | 10.65 | 0.059 | Borderline (not rejected at 5%) |
| 10 | 73.40 | 9.7 × 10⁻¹² | **Strongly rejected** |
| 20 | 94.68 | 1.1 × 10⁻¹¹ | **Strongly rejected** |

**Jarque-Bera test** checks normality using skewness and excess kurtosis:

$$JB = \frac{n}{6}\left(S^2 + \frac{(K-3)^2}{4}\right)$$

Result: JB = 4,837.3, p ≈ 0, skewness = −0.593, excess kurtosis = 8.77 − 3 = 5.77.

**What to say:**

> "The Q-Q plot shows the residuals departing from the Gaussian reference line in both tails — this is the visual signature of a heavy-tailed, leptokurtic distribution. The kurtosis of 8.77 is nearly three times the normal value of 3. The Ljung-Box test at lag 5 is borderline at p=0.059, but by lags 10 and 20 it strongly rejects white-noise residuals.
>
> This is exactly what we expect from a linear model on financial returns. The conditional mean model captures part of the predictable structure. But it leaves volatility clustering — periods where large shocks cluster together — completely unmodeled. The conditional variance is assumed constant (σ²), but financial returns violate this assumption spectacularly. That is why Phase 4 is necessary."

**Visual:** Open `VisualAnalytics.ipynb`, Block 5, the 2×2 residual diagnostic panel.

---

## SLIDE 14 — Phase 4: GARCH Volatility Modeling

**Core message:** GARCH does not predict the sign of the next return. It predicts the width of the distribution — the conditional risk around the mean. This is a fundamentally different forecasting task.

**The mean/variance separation:**

- Phase 3 asked: "What is $E[r_t | \mathcal{F}_{t-1}]$?" (conditional mean)
- Phase 4 asks: "What is $\text{Var}[r_t | \mathcal{F}_{t-1}]$?" (conditional variance)

**GARCH(1,1) with Student-t innovations:**

Let $\varepsilon_t$ be the Phase 3 residual. The conditional variance model is:

$$\sigma_t^2 = \omega + \alpha\varepsilon_{t-1}^2 + \beta\sigma_{t-1}^2$$

Each term has a precise interpretation:
- $\omega > 0$: the long-run variance floor — volatility cannot collapse to zero
- $\alpha > 0$: how strongly new shocks move current variance (shock impact)
- $\beta > 0$: how persistent variance is — how slowly it decays after a shock

**Why Student-t innovations:**

The Phase 3 Jarque-Bera result (kurtosis = 8.77) showed heavy tails. A Gaussian GARCH would under-weight extreme events. The Student-t distribution has a degrees-of-freedom parameter $\nu$: smaller $\nu$ means heavier tails, and as $\nu \to \infty$ it converges to Gaussian.

**Numerical scaling:**

The model is fit on residuals multiplied by 100 (i.e., percent units). This improves numerical stability in the optimization without changing any economics. Outputs are back-transformed to original scale.

---

## SLIDE 15 — Phase 4: GARCH Empirical Results

**Fitted parameters (all highly significant):**

| Parameter | Estimate | Std error | t-stat | p-value |
| --- | --- | --- | --- | --- |
| $\omega$ | 0.02733 | 0.00730 | 3.75 | 0.000180 |
| $\alpha$ (shock impact) | **0.12767** | 0.01616 | 7.90 | 2.8 × 10⁻¹⁵ |
| $\beta$ (persistence) | **0.86396** | 0.01556 | 55.54 | ~0 |
| $\nu$ (tail heaviness) | **5.891** | 0.580 | 10.15 | 3.3 × 10⁻²⁴ |

**Derived quantities:**

$$\text{Persistence} = \alpha + \beta = 0.1277 + 0.8640 = \mathbf{0.9916}$$

$$\text{Half-life} = \frac{\log(0.5)}{\log(\alpha + \beta)} = \frac{\log(0.5)}{\log(0.9916)} \approx \mathbf{82.4 \text{ trading days}}$$

$$\text{Unconditional volatility} = \sqrt{\frac{\omega}{1 - \alpha - \beta}} = \sqrt{\frac{0.0273}{0.0084}} \approx \mathbf{1.81\%}$$

**Diagnostic improvement — the key result:**

| Test | Before GARCH | After GARCH |
| --- | --- | --- |
| LB on squared residuals, lag 5 | Significant (autocorrelated) | p = 0.933 — **not significant** |
| LB on squared residuals, lag 10 | Significant | p = 0.633 — **not significant** |
| LB on squared residuals, lag 20 | Significant | p = 0.249 — **not significant** |
| ARCH LM test, lag 10 | Significant | p = 0.631 — **not significant** |

**What to say:**

> "Persistence of 0.9916 is very close to 1, which means volatility shocks decay very slowly. A half-life of 82 trading days means that if the market is shocked into a high-volatility regime today, more than three months will pass before the expected volatility reverts halfway to its long-run level. This is economically realistic — after events like the 2020 COVID crash or the 2022 inflation shock, elevated volatility persisted for months.
>
> The degrees-of-freedom parameter ν = 5.89 confirms heavy tails. At ν = 5.89, the Student-t distribution has finite kurtosis but substantially heavier tails than a Gaussian — consistent with the kurtosis of 8.77 we measured in the residuals.
>
> The critical diagnostic result: before GARCH, the Ljung-Box test on squared residuals was highly significant — clear evidence of volatility clustering. After GARCH filtering, the Ljung-Box on squared standardized residuals is not significant at any tested lag. The GARCH model has successfully absorbed the volatility clustering. The standardized residuals still show some structure, but the variance dynamics have been captured."

**Visual:** Open `VisualAnalytics.ipynb`, Block 6, the Ljung-Box panel before and after GARCH.

---

## SLIDE 16 — Phase 5: PatchTST-Style Transformer

**Core message:** Instead of adding AR/MA terms to the mean equation, Phase 5 tests whether a modern sequence model can find nonlinear and cross-series patterns that the linear model cannot represent.

**The fundamental SARIMAX limitation:**

SARIMAX is linear in its predictors:
$$y_t = c + \boldsymbol{\beta}^\top \mathbf{x}_t + \text{linear ARMA} + \varepsilon_t$$

It cannot capture:
- Nonlinear interactions between features (e.g., RSI × MACD momentum regime)
- Complex local temporal motifs (e.g., a specific 10-day sequence that precedes a move)
- Cross-series dependencies that change with market regime

**Why a Transformer, and why patching specifically:**

A naive transformer would treat each of the 60 past days as one token — 60 tokens × 33 features per token. PatchTST groups consecutive days into overlapping segments ("patches") before encoding them:

$$N_{patch} = 1 + \frac{L - P}{S} = 1 + \frac{60 - 10}{5} = \mathbf{11 \text{ patches}}$$

where $L = 60$ (lookback), $P = 10$ (patch length), $S = 5$ (stride).

Each 10-day segment is embedded into a latent vector and processed by self-attention. The attention mechanism decides which historical segments are most informative for the next step:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

**This architecture is explicitly grounded in the PatchTST paper:**

> Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2022). *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*. arXiv:2211.14730.

Our implementation is a CPU-friendly adaptation that keeps the two core ideas — temporal patching and shared-channel encoding — while scaling down to one-step-ahead forecasting on our financial dataset.

---

## SLIDE 17 — Phase 5: Architecture and Training

**Architecture summary:**

```
Input: X_t ∈ ℝ^{60 × 33}  (60 days × 33 features)
  ↓
Patch: unfold into 11 patches of length 10
  ↓
Embed: W ∈ ℝ^{32 × 10}   each patch → 32-dim vector
  ↓
+ Positional embedding
  ↓
Transformer encoder: 2 layers, 4 heads, FFN dim 64
  ↓
Pool across patches per channel
  ↓
Flatten + Linear head → ŷ_t (next-day NASDAQ return)
```

**Key hyperparameters:**

| Parameter | Value | Reason |
| --- | --- | --- |
| Lookback window | 60 days | ~3 trading months of context |
| Patch length | 10 days | ~2 trading weeks per token |
| Stride | 5 days | 50% overlap — patches share information |
| d_model | 32 | Small enough to train on CPU |
| Attention heads | 4 | 8-dim subspace per head |
| Transformer layers | 2 | Sufficient depth for this scale |
| Dropout | 0.10 | Light regularization |
| Optimizer | AdamW, lr=0.001, wd=0.0001 | Standard for transformers |
| Early stopping patience | 8 epochs | Best epoch reached: 9 |

**33 input features (all lagged by 1 day):**

- 7 asset log returns (SP500, NASDAQ, gold, silver, oil, platinum, palladium)
- 7 RSI(14) values
- 7 MACD histograms
- 7 Bollinger z-scores
- 2 FX log returns (EUR/USD, USD/CHF)
- 3 macro changes (GDP growth, CPI inflation, rate change)

**What to say:**

> "The model stopped improving on the validation set after epoch 9. This is the standard signature of overfitting in noisy financial data — the training loss continues to fall, but the validation loss levels off. Early stopping is essential here. Without it, the model would memorize training-sample noise rather than learning generalizable patterns."

**Visual:** Open `VisualAnalytics.ipynb`, Block 7, training curve showing train vs validation loss per epoch.

---

## SLIDE 18 — Phase 5: Results vs SARIMAX

**Performance comparison on the fixed test split (2023-10-17 to 2024-10-18):**

| Metric | SARIMAX (Phase 3) | PatchTST (Phase 5) | Change |
| --- | --- | --- | --- |
| RMSE | 0.01103 | **0.01090** | −1.2% |
| MAE | 0.00824 | **0.00810** | −1.7% |
| Directional accuracy | 52.78% | **57.14%** | +4.4 pp |
| Forecast/actual correlation | — | **0.127** | — |

**What to say:**

> "The deep model improves all three metrics, but the improvement is modest — not dramatic. This is the honest and realistic result for daily financial return prediction. The signal-to-noise ratio in daily equity returns is very low. Even a correctly specified model can only explain a small fraction of the realized next-day move because so much of it is driven by new information — macro surprises, geopolitical events, earnings announcements — that no model trained on historical data can anticipate.
>
> The directional accuracy improvement from 52.78% to 57.14% is economically meaningful if it is stable. A model that consistently gets the sign right more than half the time has potential trading value. But 'consistent' is the key word — that needs to be tested across multiple historical periods, which is exactly what Phase 6 does.
>
> The forecast distribution is also compressed around zero. The model issues conservative estimates close to the conditional mean rather than large-magnitude calls. This is expected under MSE training: the model is penalized heavily for large errors, so it hedges toward the mean."

**Visual:** Open `VisualAnalytics.ipynb`, Block 7, the three-panel comparison: forecast vs actual, distribution comparison, and rolling directional accuracy.

---

## SLIDE 19 — Phase 6: Rolling Walk-Forward Backtest Design

**Core message:** A single train/test split is insufficient. Phase 6 tests whether forecasting gains survive repeated retraining through history under transaction costs.

**Why a single split is not enough:**

A fixed split tests performance on exactly one market regime. If the test period happens to be favorable to the model (e.g., a strong trending period), the result overstates the model's general usefulness. The question we really need to answer is:

> "If we had been using this model historically, retrained it periodically with only the information available at each point in time, and paid real trading costs, how would it have performed?"

**Rolling window design:**

| Parameter | Value |
| --- | --- |
| Training window | 2,000 observations |
| Validation window | 252 observations |
| Test window | 252 observations |
| Step size | 252 observations (1 trading year) |
| Number of folds | 5 |

Fold 1 → train on earliest 2,000 days, test on the next year  
Fold 2 → shift everything forward 252 days, retrain, test on the following year  
...and so on.

**Trading rule:**

$$\text{position}_t = \text{sign}(\hat{r}_t) \in \{-1, 0, +1\}$$

A positive forecast → long position (+1). A negative forecast → short position (−1).

**Market frictions:**

$$\text{cost}_t = |\text{position}_t - \text{position}_{t-1}| \times c, \quad c = 0.0005 \text{ (5 bps)}$$

This represents 2 bps commission + 3 bps slippage per unit of position change.

$$r_t^{net} = r_t^{gross} - \text{cost}_t$$

---

## SLIDE 20 — Phase 6: Results and the Central Finding

**Overall prediction metrics (aggregated across all 5 × 252 out-of-sample days):**

| Metric | SARIMAX | PatchTST |
| --- | --- | --- |
| RMSE | 0.01612 | **0.01597** |
| MAE | 0.01152 | **0.01123** |
| Directional accuracy | 51.83% | **53.49%** |

**Trading performance after costs:**

| Financial KPI | SARIMAX | PatchTST |
| --- | --- | --- |
| Average daily turnover | 0.687 | **0.226** |
| Gross cumulative return | **+155.6%** | +83.9% |
| Net cumulative return | **+65.8%** | +59.5% |
| Annualized Sharpe ratio | **0.525** | 0.496 |
| Maximum drawdown | **−40.4%** | −42.6% |

**Fold-by-fold directional accuracy:**

| Fold | SARIMAX | PatchTST |
| --- | --- | --- |
| 1 | **61.5%** | 58.3% |
| 2 | **56.3%** | 50.0% |
| 3 | 47.6% | **54.0%** |
| 4 | 50.0% | **49.6%** |
| 5 | 43.7% | **55.6%** |

**The central finding:**

> PatchTST is statistically superior (better RMSE, MAE, hit rate). But SARIMAX is economically superior (higher Sharpe, higher net return, smaller drawdown).

**What to say:**

> "This is the most important result of the entire project — and it is a genuinely interesting one. Better statistical metrics do not automatically produce better trading performance. There are four reasons why this can happen.
>
> First, the forecast improvement is tiny relative to market noise. The RMSE difference is 0.00015 — that is the 15th decimal place of a return. On a day-to-day basis, the signal is too weak to consistently translate into better positioned trades.
>
> Second, the improvement may not occur on the days that matter most economically — the large-move days where getting the direction right generates big profits.
>
> Third, the PatchTST forecasts are more conservative in amplitude, which generates lower turnover (0.226 vs 0.687) but also lower gross returns (83.9% vs 155.6%). The lower costs of PatchTST do not fully compensate for its lower gross signal.
>
> Fourth, directional accuracy is regime-dependent. Look at the fold table: neither model dominates every fold. Fold 1 favors SARIMAX strongly. Folds 3 and 5 favor PatchTST. This is the clearest possible evidence of regime sensitivity.
>
> The honest conclusion is: we need both models. The classical benchmark remains economically competitive. The deep model shows statistical promise but needs further refinement — perhaps regime-adaptive retraining or better calibration of forecast amplitude — before it consistently dominates economically."

**Visual:** Open `VisualAnalytics.ipynb`, Block 8, cumulative wealth paths and fold-by-fold directional accuracy bar chart.

---

## SLIDE 21 — LSTM + Chronos Supplement

**Core message:** A separate notebook implements a Hybrid STL-LSTM targeting the stochastic residual, combined with Amazon Chronos zero-shot validation. The key design insight is decompose-then-predict.

**The hybrid architecture:**

STL decomposes the log-price: $\log P_t = T_t + S_t + R_t$

The LSTM is trained only on $R_t$ — the hard stochastic part. The final reconstruction is:

$$\hat{\log P}_t = T_t + S_t + \hat{R}_t^{LSTM}$$

This is principled: the statistical model handles smooth deterministic structure; the LSTM focuses all its capacity on the irregular shock component.

**LSTM architecture:**

- 2 stacked layers, hidden size 64, dropout 0.1
- Sequence length: 30 days
- 50 training epochs, gradient clipping at max_norm = 1.0
- Training loss: 0.961 → 0.792 (monotonically decreasing)

**Features:** Phase 1 OHLCV + macro + FX + Phase 2 STL components + Phase 4 GARCH variance forecasts — the richest feature set in the project, including the GARCH estimate as a live risk-regime signal.

**Amazon Chronos-T5-base zero-shot benchmark:**

- Pre-trained on millions of diverse time series
- Input: full historical log-price as univariate context
- Output: 20 sample trajectories, 5-step-ahead, 80% prediction interval (Q10/Q90)
- No task-specific training — pure zero-shot inference

**Three-way volatility comparison (last 5 days):**

| Model | Volatility proxy | MAE vs realized | Nature |
| --- | --- | --- | --- |
| GARCH(1,1) | $\hat{\sigma}_t$ from Phase 4 | Moderate | Proper conditional variance |
| Hybrid LSTM | $|\hat{\varepsilon}_t|$ (abs predicted residual) | **Lowest** | Heuristic shock-magnitude proxy |
| Chronos | $Q_{90} - Q_{10}$ (interval width) | Highest | Structural epistemic uncertainty |

**Important caveats to state explicitly:**

1. The LSTM targets **SP500 residuals**, not NASDAQ returns. Metrics are not directly comparable to Phases 3–6.
2. The 5-day evaluation window is **statistically fragile** — rankings could reverse with a different window. This is directional evidence, not a definitive ranking.
3. The Chronos interval width measures structural uncertainty over 5 days, not day-to-day conditional variance — it is not the same quantity as GARCH σ̂_t.

**What to say:**

> "Chronos is not failing when it shows the highest volatility error. It is answering a different question. GARCH asks: 'how wide is the distribution tomorrow?' Chronos asks: 'across all plausible trajectories for the next 5 days, what is the spread of outcomes?' These are related but not identical. A very calm GARCH day can still have a wide Chronos interval if the structural uncertainty is high. This is why Chronos is most valuable as a strategic uncertainty quantifier — not as a day-to-day risk model."

---

## SLIDE 22 — Overall Conclusions

**What the project demonstrated, phase by phase:**

| Phase | What we learned |
| --- | --- |
| Phase 1 | Log returns are stationary; raw prices are not. Technical indicators encode momentum, trend, and mean-reversion signals. The mixed-frequency panel requires careful alignment. |
| Phase 2 | These markets are trend-dominated with negligible stable weekly seasonality. Models should not add seasonal terms by default. |
| Phase 3 | The classical linear baseline captures some mean structure, but once returns and lagged exogenous variables are in place, no extra ARMA terms are justified. Residuals still show autocorrelation and heavy tails. |
| Phase 4 | GARCH(1,1) successfully models the volatility clustering in the residuals. Persistence of 0.9916 and half-life of 82 days — volatility shocks are long-lasting. After filtering, squared standardized residuals show no significant autocorrelation. |
| Phase 5 | The deep model improves all three metrics over the classical baseline on the single holdout. Improvement is real but modest. The signal-to-noise ratio in daily returns is low. |
| Phase 6 | Better statistical metrics do not automatically produce better trading performance. The classical benchmark remains economically competitive under repeated historical evaluation with costs. |

**The progression of complexity was justified at every step:**

SARIMAX was not beaten by a weak or naive baseline — it is a disciplined, AIC-selected, leakage-free classical model. The fact that PatchTST only modestly outperforms it statistically and does not clearly dominate economically is a scientifically honest and rigorous result.

**Three honest limitations:**

1. The evaluation period (2023–2024) is a specific market regime. The results may look different in a bear market or a high-volatility period.
2. The sign-based trading rule is the simplest possible. A realistic strategy would use the forecast distribution (confidence intervals, GARCH volatility) to size positions proportionally rather than going 100% long or short.
3. Hyperparameter choices for PatchTST (lookback window, patch size, architecture) were not exhaustively tuned — a more thorough search might widen the gap.

---

## SLIDE 23 — Anticipated Professor Questions and Answers

---

**Q: Why not use a simple LSTM instead of a PatchTST-style model for Phase 5?**

A: LSTMs process one time step at a time and accumulate context in a fixed-size hidden state. For long sequences, they suffer from vanishing gradients and struggle to relate patterns that are far apart in time. PatchTST processes the lookback window as a set of short local segments and uses self-attention to explicitly weigh relationships between any two patches regardless of their temporal distance. This is architecturally better suited for detecting recurring local motifs — such as a drawdown-and-recovery pattern that appeared 40 days ago and is reoccurring now.

---

**Q: Why is the SARIMAX order (0,0,0) with exogenous variables and not just a plain OLS regression?**

A: Technically, a SARIMAX(0,0,0) with exogenous regressors and a constant is mathematically equivalent to an OLS regression on those regressors. The difference is the framework: SARIMAX provides the full estimation infrastructure — maximum likelihood estimation, coefficient standard errors, information criteria, residual diagnostics — all consistently computed. It also allows us to extend to higher ARMA orders if AIC had justified them. Using SARIMAX from the start keeps the pipeline extensible.

---

**Q: The GARCH model still has significant Ljung-Box on the standardized residuals (not squared). Doesn't that mean GARCH failed?**

A: No. GARCH is a variance model, not a mean model. After GARCH filtering, the Ljung-Box on **squared** standardized residuals is not significant at any lag — this confirms the volatility clustering has been absorbed. The remaining significance on the (linear) standardized residuals indicates mean structure that was not captured by Phase 3's (0,0,0) SARIMAX. A natural next step would be to revisit Phase 3 with a richer ARMA structure or to add regime-switching. The GARCH model succeeded at its intended task.

---

**Q: The LSTM comparison is only over 5 days. How can you draw conclusions from that?**

A: You are right that 5 data points cannot produce a statistically stable ranking. The directional conclusion — that the multivariate LSTM proxy shows lower MAE than GARCH and Chronos over this specific window — is directionally plausible but not generalizable. A rigorous volatility model comparison would require at least 252 out-of-sample days with proper realized variance proxies, analogous to what Phase 6 does for the mean models. This is an honest limitation of the LSTM-Chronos notebook that we acknowledge explicitly.

---

**Q: Why is the LSTM targeting SP500 residuals instead of NASDAQ residuals?**

A: The notebook reads the first available asset from the Phase 2 STL output, which is SP500 alphabetically. This is an implementation detail, not a deliberate design choice. The modeling philosophy — decompose first, then predict the residual — applies equally to any asset. For a more complete comparison with Phases 3, 5, and 6, the LSTM should be retargeted to NASDAQ residuals. As it stands, the LSTM metrics are not directly numerically comparable to the other phases.

---

**Q: How do you know the ADF test result is actually detecting non-stationarity and not just being sensitive to the long sample?**

A: ADF lag selection uses AIC, which penalizes over-parameterized lag structures. The test results are robust: ADF statistics for price levels are positive or only mildly negative (+1.44 for NASDAQ, +1.20 for SP500), which is the signature of a near-unit-root or unit-root process. For log returns, the statistics are between -11 and -61 — far beyond any critical value. With 3,600+ observations, the test has very high power, and the magnitude of the statistics for returns (not just the p-value) confirms the conclusion.

---

**Q: Why 5 basis points of transaction costs? Is that realistic?**

A: For liquid large-cap indices and ETFs, 2–5 bps round-trip is a reasonable lower bound for institutional trading. Retail trading would be higher. We use 5 bps (2 commission + 3 slippage) as a conservative but realistic estimate for a strategy that trades daily. The critical point is that even at this level, the SARIMAX strategy's net return falls from 155.6% gross to 65.8% net — a reduction of nearly 60% of gross return. This illustrates why turnover is a first-order concern in systematic strategies, not a secondary consideration.

---

**Q: Could you combine GARCH volatility forecasts with the PatchTST mean forecasts into a unified strategy?**

A: Yes — this is the natural next step. A combined model would use the PatchTST directional signal for position sizing and the GARCH conditional volatility to scale the position: larger position when volatility is low (lower risk), smaller position when volatility is high (the forecast may be correct but the realized move is uncertain). This is known as a volatility-scaled or risk-parity approach and is standard in quantitative finance. It would likely improve the Sharpe ratio relative to the naive sign-based strategy implemented in Phase 6.

---

## APPENDIX — Complete Numbers Reference

**Dataset:**
- Raw rows: 3,904 | Trimmed pre-2010: 55 | Non-trading removed: 183 | Cleaned: 3,666 | Modeling: 3,605

**Phase 3 train/test split:**
- Train: 3,351 (2010-05-03 to 2023-10-16) | Test: 252 (2023-10-17 to 2024-10-18)

**Phase 3 AIC:** auto_arima = −19,559.6 | SARIMAX refit = −19,553.5

**Phase 4 unconditional volatility:** $\sqrt{\omega / (1 - \alpha - \beta)} = \sqrt{0.0273 / 0.0084} = 1.81\%$

**Phase 5 validation directional accuracy:** 48.4% — lower than test (57.1%), confirming regime dependence

**Phase 6 fold-level notes:**
- Fold 4: SARIMAX selected AR(1) instead of (0,0,0) — the only fold where auto_arima chose a non-trivial order
- Folds 3, 5: PatchTST clearly wins directional accuracy; Folds 1, 2: SARIMAX wins — no model dominates consistently

**Chronos model used:** `chronos-t5-base` (not Chronos-2), loaded via `ChronosPipeline.from_pretrained()` with `device_map="cpu"`, `dtype=torch.float32`
