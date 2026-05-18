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

## EDA Notebook Deep Dive

The `Assignment1/01_EDA.ipynb` notebook is the exploratory foundation that justifies every subsequent modeling decision. It answers five fundamental questions about the raw dataset before a single model is fit.

### 1. Mixed-Frequency Identification

The notebook computes inter-observation gaps for the three macro variables and reports them explicitly:

- **GDP**: 57 non-null entries, mean gap of approximately 91 days. This confirms the data is quarterly. The GDP series starts at 2010-04-01, which is the primary reason the dataset is trimmed to that date.
- **CPI**: 176 non-null entries, mean gap of approximately 30-31 days. Monthly frequency confirmed.
- **US federal funds rate**: 176 non-null entries, same gap structure as CPI. Monthly frequency confirmed.

This is not a cosmetic check. The update frequency of each variable determines the only defensible imputation strategy. Quarterly GDP values should be held constant until the next quarterly release; they should not be interpolated as if they were daily measurements.

### 2. Data Quality Validation

After trimming to 2010-04-01 and forward-filling the three macro variables, a null check confirms zero remaining NaNs for GDP, CPI, and us_rates_%. The first 20 rows of the cleaned frame verify that the 2010-04-01 quarterly GDP reading propagates daily until the next release arrives.

The `.describe()` output on the full numeric frame confirms two important data integrity properties:

- **No negative prices**: minimum values for all close columns are strictly positive, which eliminates any risk of undefined log-return computation (log of zero or negative numbers).
- **Interpretable max/min ratios**: the NASDAQ close ratio is approximately 5.7, consistent with the observed equity bull market from 2010 to 2024. The oil volume ratio is approximately 1360, driven by the April 2020 negative-price episode and its volume spike, which is economically interpretable rather than indicative of data corruption.

### 3. Change Frequency Analysis

By counting how many times each column changes its value across the entire frame, the notebook produces a quantitative confirmation of the update structure:

- GDP: 57 changes (quarterly)
- US rates: 119 changes (slightly more than quarterly × 2 because the FOMC can change rates at scheduled meetings between quarterly GDP prints)
- CPI: 173 changes (approximately monthly)
- Market OHLCV columns: approximately 3836–3849 changes, consistent with daily market updates

This analysis independently verifies the forward-fill assumption without relying on the calendar structure of the raw dates alone.

### 4. Distribution and Outlier Analysis

KDE plots of close prices across the seven assets reveal:

- **Equity indices** (SP500, NASDAQ): positively skewed unimodal distributions reflecting the long bull trend
- **Commodities** (oil, silver, palladium): multi-modal distributions or wider spreads, reflecting the more volatile price behavior of commodity markets over this period
- **Gold**: relatively symmetric distribution with moderate skew, consistent with gold's safe-haven role and lower realized volatility compared to silver or palladium

Log-return KDEs show all seven assets centered near zero with visible fat tails in both directions. Oil has the widest spread due to the 2020 crash episode. Boxplots confirm that every asset has statistical outliers (dots beyond the whiskers), which is the empirical basis for using heavy-tailed distributions (Student-t) in GARCH estimation.

### 5. Normalized Price Trajectories

The close prices for all seven assets are placed on a common scale using a **base-100 index**, also called price rebasing. The formula is:

$$\text{Index}_t = \frac{P_t}{P_{t_0}} \times 100$$

where:

- $P_t$ is the close price of the asset on trading day $t$
- $P_{t_0}$ is the close price of the **same asset** on the first cleaned trading day (2010-04-01, the base date)
- the result equals exactly 100 for every asset on the base date

This is implemented directly as:

```python
normalized = cleaned_df.set_index("date")[close_columns].div(
    cleaned_df.set_index("date")[close_columns].iloc[0]
) * 100
```

The `.iloc[0]` selects the first row of the cleaned dataframe. Each column is divided independently by its own first value, so each asset starts at 100 regardless of its original price level.

**This has nothing to do with moving averages, rolling windows, or any momentum calculation.** It is a one-time division by a fixed base-date value applied to the entire price series. The only parameters are:

- the base date (implicitly 2010-04-01, the start of the cleaned dataset)
- the scale factor (100, for readability)

**How to read the result:** an index value of 200 means the asset has exactly doubled from its 2010-04-01 price. A value of 50 means it has halved. A value of 580 (approximately where NASDAQ reaches by 2024) means it has grown to 5.8 times its starting price.

**Why this normalization is used instead of raw prices:** NASDAQ close was approximately 2,530 points at the start of the sample, while gold was approximately 1,180 USD/oz and silver approximately 17.5 USD/oz. Plotting these on a single y-axis would compress the lower-priced series into a flat line at the bottom. The base-100 index removes the level difference and makes percentage growth directly comparable across all assets.

The key observations from this visualization:

- **NASDAQ** delivers the strongest cumulative return over the period, approximately 580-indexed (5.8× from the base)
- **SP500** is the second strongest performer at approximately 480-indexed
- **Gold** shows a positive but moderate trajectory
- **Oil, silver, platinum, palladium**: exhibit mean-reverting or structurally volatile patterns without a sustained positive trend comparable to equity indices

This cross-asset comparison provides empirical justification for choosing NASDAQ as the primary forecasting target. NASDAQ has the richest trend signal, the highest realized return, and represents one of the most economically important indices in the dataset.

### 6. Volume and Intraday Spread Behavior

Volume time series plotted per asset confirm that:

- Market volumes exhibit regime changes (notably the 2020 COVID volatility period)
- High-Low spread (a proxy for intraday volatility) spikes sharply during crisis periods

The high-low spread analysis is a useful complement to log-return volatility because it captures intraday risk rather than only day-to-day price changes. These patterns foreshadow the volatility clustering that GARCH will later formalize.

### 7. Macro Events Overlay

The three macro series (GDP, CPI, us_rates_%) are plotted with vertical markers at known market-stress dates:

- **2020-03-20**: COVID crash peak → visible GDP drop and rate-cut response
- **2022-03-16**: Fed rate hike cycle start → visible CPI peak and rapid rate normalization

This visualization contextualizes the macro data and helps explain why macro changes are included as exogenous predictors in the SARIMAX and PatchTST models.

### 8. Correlation Structure

A lower-triangular heatmap of close prices plus macro and FX variables reveals:

- Very high positive correlations between SP500 and NASDAQ (~0.99 at price levels)
- Moderate positive correlations between equity indices and gold
- Near-zero or weakly negative correlations between equities and oil at price levels
- EUR/USD and USD/CHF show moderate negative correlation with each other (reflecting inverse CHF safe-haven dynamics against USD)

A key caveat: these correlations are computed at price **levels**, which are non-stationary. They reflect shared trending rather than direct economic dependence. The relevant correlations for modeling are those computed at the **return** level, which are substantially lower and used in the feature design for Phases 3 and 5.

## Data Transformations: Initial Rationale

### Raw prices vs Returns

Raw prices are usually non-stationary: their mean and variance structure drift (change) over time and generally exhibit exponential growth trajectories over long horizons, which breaks the assumptions behind ARIMA-class models and weakens statistical inference. Therefore, that's why we choose returns.

Simple returns measure the straightforward percentage change between two periods and are calculated as the difference between the current and previous price divided by the previous price. While intuitive and asset-additive—meaning the return of a portfolio is the weighted sum of the simple returns of its constituent assets—they present significant statistical drawbacks for time series modeling. Simple returns are bounded below by -100%, as an asset's price cannot drop below zero, but they are theoretically unbounded on the upside, leading to an asymmetric distribution. Furthermore, the product of normally distributed variables does not yield a normal distribution, complicating probabilistic modeling.

### Why log returns instead of simple returns

Logarithmic returns, calculated as the natural logarithm of the ratio of the current price to the previous price, are the standard for advanced time series forecasting. There are three distinct mathematical reasons for this.

#### Reason 1 — Additivity over time

Log returns add across time periods while simple returns do not. For a sequence of periods:

$$r_{t_1 \to t_3} = \log\!\left(\frac{S_{t_3}}{S_{t_1}}\right) = \log\!\left(\frac{S_{t_2}}{S_{t_1}}\right) + \log\!\left(\frac{S_{t_3}}{S_{t_2}}\right) = r_{t_1 \to t_2} + r_{t_2 \to t_3}$$

The cumulative log return over any interval is just the sum of the sub-period log returns. Simple returns do not have this property — they compound multiplicatively: $(1 + R_{t_1 \to t_3}) = (1 + R_{t_1 \to t_2})(1 + R_{t_2 \to t_3})$.

Additivity matters for modeling: summing independent random variables and applying standard regression or time-series tools is far more tractable than working with products.

#### Reason 2 — Symmetry of gains and losses

With simple returns, a 50% loss followed by a 100% gain brings you back to the starting point (0.5 × 2.0 = 1.0), but the magnitudes look asymmetric (+100% vs −50%). With log returns, the same round-trip is $\log(0.5) + \log(2) = -0.693 + 0.693 = 0$, perfectly symmetric. This symmetry makes log returns easier to model with symmetric distributions like the Gaussian or Student-t.

#### Reason 3 — Domain alignment: prices are positive, so log returns live on all of ℝ

Prices must be strictly positive: $S_t > 0$ always. Therefore $S_T / S_0 \in (0, +\infty)$. The natural logarithm maps $(0, +\infty)$ **bijectively** to $(-\infty, +\infty) = \mathbb{R}$:

| Price ratio $S_T/S_0$ | Log return | Economic meaning |
| --- | --- | --- |
| $\to 0^+$ | $\to -\infty$ | Near-total loss |
| $0.5$ | $-0.693$ | Price halved |
| $1.0$ | $0$ | No change |
| $2.0$ | $+0.693$ | Price doubled |
| $\to +\infty$ | $\to +\infty$ | Unlimited gain |

Compare this to the **simple return** $R = S_T/S_0 - 1$, which maps $(0, +\infty)$ to $(-1, +\infty)$ — simple returns are bounded below by $-1$ (losing more than 100% is impossible). A Gaussian distribution places positive probability mass on values below $-1$, which simple returns can never achieve. The domain is inconsistent.

Log returns live on all of $\mathbb{R}$, exactly matching the support of Gaussian (and Student-t) error distributions. This is why regression models and ARIMA-class models with Gaussian errors are domain-consistent when the target is a log return but not when it is a simple return.

#### Geometric Brownian Motion — the theoretical justification

The domain argument above is necessary but not sufficient. The stronger theoretical justification comes from the **Geometric Brownian Motion (GBM)** model, which is the canonical continuous-time model for asset prices.

**What GBM is.** A standard Brownian motion $W_t$ is a random process where increments $W_t - W_s \sim \mathcal{N}(0, t-s)$ are independent and Gaussian — it is accumulated Gaussian noise through time. GBM models the price $S_t$ via the stochastic differential equation (SDE):

$$dS_t = \mu S_t\,dt + \sigma S_t\,dW_t$$

- $\mu S_t\,dt$: deterministic drift proportional to the current price (μ = expected return per unit time)
- $\sigma S_t\,dW_t$: random shock proportional to the current price (σ = volatility)

Both terms are proportional to $S_t$, which is what makes it *geometric*. This proportionality ensures prices stay positive and that percentage volatility (not absolute dollar volatility) is constant — a 1% move on a $1,000 stock is the same model-magnitude as a 1% move on a $10 stock.

**Why log returns are Gaussian under GBM.** Apply **Itô's lemma** (the stochastic calculus chain rule for nonlinear functions of Brownian processes) to $f(S_t) = \log S_t$:

$$d(\log S_t) = \frac{1}{S_t}\,dS_t - \frac{1}{2}\cdot\frac{1}{S_t^2}\cdot(dS_t)^2$$

The second term is the Itô correction that standard calculus does not have (it arises because $W_t$ has non-zero quadratic variation: $(dW_t)^2 = dt$ in the Itô sense). Substituting $dS_t = \mu S_t\,dt + \sigma S_t\,dW_t$ and using $(dS_t)^2 = \sigma^2 S_t^2\,dt$:

$$d(\log S_t) = \frac{\mu S_t\,dt + \sigma S_t\,dW_t}{S_t} - \frac{\sigma^2 S_t^2\,dt}{2S_t^2} = \left(\mu - \frac{\sigma^2}{2}\right)dt + \sigma\,dW_t$$

The right-hand side now has constant coefficients — no $S_t$ anywhere. Integrating from $0$ to $T$:

$$\log\!\left(\frac{S_T}{S_0}\right) = \left(\mu - \frac{\sigma^2}{2}\right)T + \sigma W_T$$

Since $W_T \sim \mathcal{N}(0, T)$:

$$\boxed{\log\!\left(\frac{S_T}{S_0}\right) \sim \mathcal{N}\!\!\left(\left(\mu - \frac{\sigma^2}{2}\right)T,\; \sigma^2 T\right)}$$

**The log return is a linear function of a Gaussian random variable — therefore it is Gaussian under GBM.** This is the formal justification for the sentence "if asset prices follow a geometric Brownian motion, their log returns are normally distributed."

**The Itô correction $\sigma^2/2$.** The expected log return $(\mu - \sigma^2/2)T$ is always slightly less than $\mu T$. This is not an approximation — it is exact, and it reflects the mathematics of compounding: variance in returns always erodes geometric growth relative to arithmetic growth. A strategy with drift $\mu = 0.10$ and volatility $\sigma = 0.20$ has expected log return $(0.10 - 0.02)T = 0.08T$, not $0.10T$. This gap is the **variance drag** and it is real and important in portfolio management.

#### The honest caveat — GBM is a theoretical model, not an empirical fact

GBM gives the theoretical motivation for log returns, but it makes three assumptions that are empirically violated by real financial data:

1. **Normally distributed log returns**: real log returns have fat tails (kurtosis >> 3) and mild negative skew. The Jarque-Bera test in Phase 3 strongly rejects Gaussianity. The Q-Q plots show clear tail departures. This is why Phase 4 uses a Student-t innovation distribution for GARCH.

2. **i.i.d. returns across time**: real log returns show volatility clustering — large moves tend to be followed by large moves regardless of sign. This is the entire empirical motivation for GARCH: if returns were i.i.d., GARCH would have nothing to model.

3. **Constant variance**: the variance $\sigma^2$ is assumed fixed in GBM but varies over time in reality. GARCH explicitly models this time-variation.

The correct framing is: **GBM provides the theoretical justification for working with log returns** (domain, additivity, Gaussian ideal), while the subsequent modeling choices (GARCH, Student-t, nonlinear features) acknowledge and correct for the ways real markets deviate from GBM. Log returns are not chosen because markets follow GBM — they are chosen because log returns have the right mathematical properties regardless of whether GBM holds exactly.

For a price series $(P_t)$, the simple return is:

$$
R_t = \frac{P_t - P_{t-1}}{P_{t-1}}
$$

and the log return is

$$
r_t = \log\left(\frac{P_t}{P_{t-1}}\right) = \log(1 + R_t).
$$

Log returns are usually preferred in quantitative modeling because:

- they are additive over time, which simplifies aggregation and modeling,
- they often stabilize variance better than raw prices,
- and many financial models are formulated naturally in continuously compounded returns.

For small returns, log returns and simple returns are numerically close, since $\log(1+R_t) \approx R_t$

For finance, one subtle point matters a lot: prices often behave more like a multiplicative process than an additive one. In plain words, a 2% move when oil is at 100 is not the same size in dollars as a 2% move when oil is at 40. That is why I decomposed log prices, not raw prices.

If roughly:

$$
P_t \approx T_t \times S_t \times E_t
$$

then after taking logs:

$$
\log P_t \approx \log T_t + \log S_t + \log E_t
$$

Now the decomposition becomes additive, which is exactly what STL wants.

### Missing-data handling

The current notebooks use forward fill for macroeconomic variables after removing the earliest rows with missing GDP. This is defensible because macroeconomic indicators are published discretely and remain the latest known value until the next release. We will preserve this logic, but is a mixed-frequency alignment assumption rather than a generic imputation trick.

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

### Explicit statement of the benchmark model

The benchmark model used in Phase 3 is **SARIMAX**, not plain ARIMA.

More precisely:

- `pmdarima.auto_arima` is used only to **search over candidate ARIMA orders** with AIC,
- the chosen order is then **refit as a `statsmodels` SARIMAX model**,
- and all reported forecasts, coefficients, residuals, and diagnostics come from that fitted SARIMAX model.

So the correct description is:

"The classical benchmark is a SARIMAX model whose ARIMA order was selected by `auto_arima`."

That means:

- `auto_arima` is the **selection procedure**,
- `SARIMAX` is the **actual fitted benchmark model**.

### Why ARIMA and SARIMAX are used as a baseline

Before moving to volatility models and deep learning, a serious time-series project should establish whether a simpler linear model already explains most of the predictable structure. ARIMA-class models are valuable because they are:

- interpretable,
- statistically grounded,
- and useful as a benchmark that later, more complex models must beat.

If a deep model performs only marginally better than a disciplined classical baseline, that is a very different result from beating a weak or poorly specified baseline.

### SARIMAX built from first principles

SARIMAX is not a single model — it is an acronym for a family of ideas stacked on top of each other. Understanding it requires building each layer from scratch.

#### Step 1 — AR(p): autoregression

The simplest time-series model is the **AutoRegressive** model of order *p*:

$$
y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \cdots + \phi_p y_{t-p} + \varepsilon_t
$$

Today's value is a weighted sum of the last *p* values plus a noise term $\varepsilon_t$. The weights $\phi_1, \ldots, \phi_p$ are estimated from data. The order *p* controls how many lags of the series enter the mean equation.

- AR(0): $y_t = c + \varepsilon_t$ — just a constant plus noise (white noise around a mean)
- AR(1): $y_t = c + \phi_1 y_{t-1} + \varepsilon_t$ — only yesterday matters
- AR(2): also includes two days ago

#### Step 2 — MA(q): moving average on shocks

The **Moving Average** model of order *q* regresses on past *forecast errors* (shocks), not on past values of the series itself:

$$
y_t = c + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \theta_2 \varepsilon_{t-2} + \cdots + \theta_q \varepsilon_{t-q}
$$

The idea: if yesterday I experienced a larger-than-expected shock, today's forecast should be adjusted by the magnitude of that shock weighted by $\theta_1$. MA terms capture the persistence of shocks in the mean equation.

Combining AR and MA gives **ARMA(p, q)**.

#### Step 3 — I(d): differencing for non-stationarity

**ARIMA(p, d, q)** — the **I** stands for Integrated. If a series has a unit root (non-stationary, like price levels), it can be differenced *d* times to make it stationary before ARMA is applied. First differencing ($d=1$) transforms $y_t$ into $y_t - y_{t-1}$.

For log-return series, the ADF test already showed the series are stationary — so $d = 0$ and no differencing is needed. The return transformation itself eliminated the non-stationarity.

#### Step 4 — SARIMA(p,d,q)(P,D,Q,m): seasonal extension

**SARIMA** adds a second set of AR, I, and MA terms that operate at multiples of a seasonal period *m* (e.g., $m = 5$ for a trading week). The outer bracket $(P, D, Q, m)$ is the seasonal counterpart of the inner bracket:

- *P*: seasonal AR order (lags at $m$, $2m$, $3m$, ...)
- *D*: seasonal differencing
- *Q*: seasonal MA order
- *m*: season length

The STL decomposition in Phase 2 showed seasonal strength ≈ 0.056 for NASDAQ — negligible. There is no stable weekly pattern to model, so $P = D = Q = 0$ and $m = 0$.

#### Step 5 — SARIMAX: add exogenous regressors (the X)

**SARIMAX** adds external variables $x_t$ to the mean equation:

$$
y_t = c + \underbrace{\phi_1 y_{t-1} + \cdots + \theta_q \varepsilon_{t-q}}_{\text{ARIMA mean}} + \underbrace{\beta_1 x_{t-1}^{(1)} + \beta_2 x_{t-1}^{(2)} + \cdots + \beta_k x_{t-1}^{(k)}}_{\text{exogenous block}} + \varepsilon_t
$$

The $\beta$ coefficients capture how lagged external signals (SP500 return, gold, oil, FX rates, GDP growth, CPI inflation, interest rate changes) shift the conditional mean of the target series beyond what its own dynamics explain.

### What each component of the fitted (0,0,0)(0,0,0,0) means

`auto_arima` selected all orders equal to zero. Here is what each zero means for this specific project:

| Component | Value | What it means here |
| --- | --- | --- |
| p = 0 | AR order | NASDAQ log returns have no statistically useful own-lagged mean structure beyond what the exogenous block already captures |
| d = 0 | Integration | Log returns are already stationary (ADF test confirmed); no differencing needed |
| q = 0 | MA order | Past forecast errors do not add predictive power to the conditional mean after the exogenous block is included |
| P = 0 | Seasonal AR | No seasonal autoregression at weekly multiples |
| D = 0 | Seasonal differencing | No seasonal unit root |
| Q = 0 | Seasonal MA | No seasonal moving-average structure |
| m = 0 | Season length | STL confirmed no stable weekly seasonality |

**Net result — the fitted mean equation is:**

$$
\hat{y}_t = c + \beta_1 \cdot x_{t-1}^{(\text{sp500})} + \beta_2 \cdot x_{t-1}^{(\text{gold})} + \beta_3 \cdot x_{t-1}^{(\text{oil})} + \beta_4 \cdot x_{t-1}^{(\text{eur\_usd})} + \beta_5 \cdot x_{t-1}^{(\text{usd\_chf})} + \beta_6 \cdot x_{t-1}^{(\text{gdp})} + \beta_7 \cdot x_{t-1}^{(\text{cpi})} + \beta_8 \cdot x_{t-1}^{(\text{rate})}
$$

This is a **linear regression on 8 lagged exogenous variables plus an intercept** — no autoregressive terms, no moving-average terms. The SARIMAX framework is used for estimation, diagnostics, and forecasting infrastructure, but the mean model it contains is equivalent in structure to OLS regression on lagged predictors.

### Why AIC selecting (0,0,0) is a meaningful empirical result

This is a meaningful empirical finding, not a model failure. An ARIMA order of (0,0,0) does **not** mean:

- the series contains no information,
- the model is trivial or useless,
- or the project failed to find structure.

It means something narrower and more precise: after transforming the target into log returns (which removed the trend), after adding 8 lagged exogenous predictors (which captured equity market, FX, and macro structure), the data does not justify adding extra AR or MA terms to the conditional mean equation under AIC. The predictable structure in the own-lag dynamics of NASDAQ returns, beyond what the exogenous block captures, is too weak to earn the extra parameters.

This is exactly what the **Efficient Market Hypothesis** predicts: in a large, liquid, heavily-watched market like NASDAQ, simple linear patterns in past returns are arbitraged away. The remaining predictable structure lives in the cross-asset linkages (captured by the exogenous block) and in the variance process (captured by GARCH in Phase 4), not in NASDAQ's own AR or MA terms.

### The SARIMAX / GARCH division of labor

The SARIMAX model in Phase 3 handles the **conditional mean** — it estimates $E[y_t | \mathcal{F}_{t-1}]$, the expected return given all available lagged information.

The GARCH model in Phase 4 handles the **conditional variance** — it estimates $\text{Var}[y_t | \mathcal{F}_{t-1}] = \sigma_t^2$, the time-varying uncertainty around that mean.

These two models are complementary, not competing. They are **not summed** — they model two different statistical moments of the same return. The SARIMAX residuals (which still contain volatility clustering, as shown by the Ljung-Box test on squared residuals at lags 10 and 20) become the input to the GARCH model. The complete two-phase model writes:

$$y_t = \hat{y}_t^{\text{SARIMAX}} + \varepsilon_t, \qquad \varepsilon_t = \sigma_t z_t, \qquad z_t \overset{\text{iid}}{\sim} t_\nu(0,1)$$

$$\sigma_t^2 = \omega + \alpha\varepsilon_{t-1}^2 + \beta\sigma_{t-1}^2$$

Together they describe the full conditional distribution of tomorrow's return:

$$y_t \mid \mathcal{F}_{t-1} \sim \text{scaled-}t_\nu\!\left(\hat{y}_t^{\text{SARIMAX}},\; \sigma_t^2\right)$$

SARIMAX provides the **center** of that distribution; GARCH provides its **width**. This is the standard two-step approach in financial econometrics: fit the mean model first, extract residuals, fit the variance model on those residuals.

#### How they are actually used in practice — and where they are not combined

Despite the theoretical joint model above, SARIMAX and GARCH are used **separately** in this project's trading evaluation:

- **Phase 6 (backtesting)**: the trading rule is $\text{position}_t = \text{sign}(\hat{y}_t^{\text{SARIMAX}})$. Only the SARIMAX mean forecast drives trading decisions. The GARCH volatility estimate $\sigma_t$ is **not used** to size or filter positions.

- **`03_Forecasting_LSTM_&_Chronos.ipynb`**: the GARCH conditional variance $\sigma_t^2$ **is** passed as an input feature to the LSTM. The LSTM can therefore condition its residual prediction on the current volatility regime. This is the closest the project comes to truly combining the two models — GARCH output feeds into the deep model's feature vector.

#### What a true joint combination would look like

A natural extension that this project does not implement is **volatility-scaled position sizing**:

$$\text{position}_t = \frac{\hat{y}_t^{\text{SARIMAX}}}{\sigma_t^{\text{GARCH}}}$$

This gives a larger position when the forecast is large *relative to current uncertainty*, and a smaller position when the market is highly volatile. It is related to the Kelly criterion and is standard in professional systematic strategies. Under this rule, GARCH actively modulates the trading signal — a high-confidence forecast during a calm period generates a large position; the same forecast during a volatility spike generates a smaller one. This would be a direct, economically motivated combination of the two models and represents a natural next step beyond the current Phase 6 implementation.

### How and where the exogenous block is built

The exogenous block is constructed entirely in one method: `SarimaxBaselineBuilder._build_design_matrix` in `Assignment1/src/advml_assignment1/phase3_classical_baseline.py` (lines 136–152). It reads the Phase 1 modeling dataset and produces the 9-column design matrix that SARIMAX receives.

The design matrix has this structure:

| Column | Role | Meaning |
| --- | --- | --- |
| `date` | index | trading day $t$ |
| `target` | dependent variable | `nasdaq log_return` on day $t$ (what we predict) |
| `sp500_ret_l1` | exogenous | S&P 500 log return on day $t-1$ |
| `gold_ret_l1` | exogenous | gold log return on day $t-1$ |
| `oil_ret_l1` | exogenous | crude oil log return on day $t-1$ |
| `eur_usd_ret_l1` | exogenous | EUR/USD log return on day $t-1$ |
| `usd_chf_ret_l1` | exogenous | USD/CHF log return on day $t-1$ |
| `gdp_growth_l1` | exogenous | log GDP growth on day $t-1$ (quarterly release) |
| `cpi_inflation_l1` | exogenous | log CPI inflation on day $t-1$ (monthly release) |
| `rate_change_l1` | exogenous | simple change in US federal funds rate on day $t-1$ |

The exact code for each column group is:

```python
# --- Group 1: market returns (sp500, gold, oil) ---
# log return is already in the Phase 1 modeling_data.csv
for asset_name in ("sp500", "gold", "oil"):
    design[f"{asset_name}_ret_l1"] = frame[f"{asset_name} log_return"].shift(1)

# --- Group 2: FX log returns ---
# FX pairs (eur_usd, usd_chf) are raw prices in Phase 1, so the log return
# is computed inline and then lagged
for fx_column in ("eur_usd", "usd_chf"):
    fx_return = np.log(frame[fx_column] / frame[fx_column].shift(1))
    design[f"{fx_column}_ret_l1"] = fx_return.shift(1)

# --- Group 3: macro changes ---
# GDP and CPI are log-differenced (growth rates) before lagging
design["gdp_growth_l1"]  = np.log(frame["GDP"] / frame["GDP"].shift(1)).shift(1)
design["cpi_inflation_l1"] = np.log(frame["CPI"] / frame["CPI"].shift(1)).shift(1)

# US rates are already in percentage-point units, so a simple first difference
# is used instead of a log difference
design["rate_change_l1"] = frame["us_rates_%"].diff().shift(1)
```

### Three different transformations for three different variable types

Each of the three groups receives a different transformation. This is not arbitrary — it is driven by the statistical properties of each variable.

#### Group 1 and 2: log return for prices and FX rates

All prices and exchange rates grow approximately multiplicatively, so the log return is the natural transformation:

$$r_t = \log\left(\frac{P_t}{P_{t-1}}\right)$$

For the three market assets (SP500, gold, oil), the log return was already computed in Phase 1 and is available as a column in the modeling dataset. For the two FX pairs (EUR/USD, USD/CHF), Phase 1 stored the raw price levels — so the log return is recomputed inline in `_build_design_matrix`.

#### Group 3a: log growth for GDP and CPI

GDP and CPI are also level variables published at low frequency (quarterly and monthly). They receive the same log-difference treatment as prices:

$$g_t = \log\left(\frac{M_t}{M_{t-1}}\right)$$

On most daily rows, this value is zero because GDP and CPI do not change between quarterly or monthly release dates (the value was forward-filled by Phase 1). On the specific dates of a new release, the log difference captures the realized growth or inflation rate for that period.

#### Group 3b: simple difference for the US policy rate

The federal funds rate is already measured in percentage-point units (e.g., 0.25 %, 5.50 %). Taking a log difference of a number that can be zero or very close to zero is numerically undefined or unstable. A simple first difference is both safer and more interpretable:

$$\Delta r_t = r_t - r_{t-1}$$

A value of +0.25 means the Fed raised rates by 25 basis points. A value of 0 means no change. A value of -0.50 means a 50-bps cut.

### Why the lag of exactly 1 day prevents leakage

Every exogenous column ends with `_l1`, which means the value seen by the model on day $t$ belongs to day $t-1$. This single `.shift(1)` is the leakage-prevention mechanism.

To make this concrete: suppose we are forecasting the NASDAQ return for Wednesday 2024-01-03.

- The target is the NASDAQ return that closes at the end of Wednesday.
- The model receives the S&P 500 return from Tuesday 2024-01-02, the EUR/USD move from Tuesday, the most recent GDP reading as of Tuesday, and so on.
- The model does **not** see the S&P 500 return from Wednesday itself.

Without the `.shift(1)`, the model would use same-day SP500 information to predict same-day NASDAQ returns. Since these two indices are nearly perfectly contemporaneously correlated, the model would appear to achieve extremely high accuracy in training — but would be completely useless in live deployment where Wednesday's SP500 return is not yet known at the time we need to make Wednesday's NASDAQ forecast.

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

$$
AIC = 2k - 2\log L
$$

where $(k)$ is the number of estimated parameters and $(L)$ is the maximized likelihood.

The idea is to reward goodness of fit while penalizing unnecessary complexity. A model with more parameters can fit the training data better almost by construction, so AIC helps prevent choosing an over-parameterized specification.

In this project, `pmdarima.auto_arima` searches candidate ARIMA orders and selects the one with the best AIC. The chosen order is then refit using `statsmodels` SARIMAX so that full diagnostics and forecasting outputs can be produced cleanly.

### Exact benchmark specification used in Phase 3

The fitted benchmark in Phase 3 is:

- model class: `SARIMAX`
- target: `nasdaq log_return`
- non-seasonal order: `(0, 0, 0)`
- seasonal order: `(0, 0, 0, 0)`
- trend: constant term, `trend='c'`
- exogenous regressors:
  - `sp500_ret_l1`
  - `gold_ret_l1`
  - `oil_ret_l1`
  - `eur_usd_ret_l1`
  - `usd_chf_ret_l1`
  - `gdp_growth_l1`
  - `cpi_inflation_l1`
  - `rate_change_l1`

These exogenous variables are standardized on the training sample before fitting.

So the benchmark is best described as:

"a non-seasonal SARIMAX return model with an intercept and lagged exogenous predictors."

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

### What the selected `(0,0,0)` model really means

It is important not to misread the selected order.

An ARIMA order of `(0,0,0)` does **not** mean:

- the series contains no information,
- the model is trivial in a useless sense,
- or there is nothing left to explain.

What it means here is narrower and more precise:

- after transforming the target into returns,
- after adding the lagged exogenous block,
- and after comparing candidate ARIMA orders with AIC,

the data does not justify adding extra autoregressive or moving-average terms to the conditional **mean** equation.

So the fitted mean model is essentially a regression-style return model with exogenous predictors and an intercept. This is a valid and informative benchmark because it tells us that the predictable structure in the mean is limited.

Another precise way to say it is:

"The model belongs to the ARIMA family, but because exogenous regressors are included, the implemented benchmark is SARIMAX. Since the selected ARIMA order is `(0,0,0)`, the final fitted mean equation behaves more like a linear regression on lagged exogenous variables plus a constant than like a richer autoregressive or moving-average return model."

### Why coefficient interpretation must be done carefully

The exogenous variables are standardized using training-sample means and standard deviations before fitting the SARIMAX model. Therefore:

- the coefficients are numerically stable for estimation,
- and the coefficient magnitudes are on a comparable scale,
- but they are **not** raw-unit effects from the original variables.

This means a coefficient should be read as:

"If this predictor increases by roughly one training-sample standard deviation, how does the conditional mean of the target return change, holding the other predictors fixed?"

This is especially important when comparing variables like:

- market returns,
- FX returns,
- GDP growth,
- and rate changes,

because these live on very different natural scales.

### Example: how to explain the NASDAQ coefficient result

The strongest significant coefficient is the lagged `sp500` return, with an estimated value of approximately `-0.001525` after standardization.

A careful classroom explanation would be:

"Conditional on the rest of the lagged exogenous block, a one-standard-deviation increase in the lagged S&P 500 return is associated with a small negative shift in the next NASDAQ log-return forecast."

What this does **not** prove:

- it does not prove a causal effect,
- it does not prove that the NASDAQ and S&P 500 move oppositely in general,
- and it does not override the well-known fact that these markets are strongly positively related contemporaneously.

What it may suggest is a modest short-horizon conditional reversal effect in the fitted linear mean equation. That is a model-based interpretation, not a universal market law.

### How to interpret the forecast visual

The forecast chart in the visual notebook should be read in layers:

1. The **training fitted line** shows how well the model explains the conditional mean in-sample.
2. The **test forecast line** shows the out-of-sample mean prediction.
3. The **confidence band** shows forecast uncertainty around that mean.
4. The difference between the actual test series and the forecast line shows what the model fails to explain.

For NASDAQ returns, the main visual lesson is that:

- the forecast line is much smoother than the realized return path,
- the realized series still exhibits sharp positive and negative shocks,
- and the model captures broad mean behavior much better than shock magnitude.

This is exactly what one should expect from a classical conditional-mean model on financial returns.

### How to interpret the coefficient visual

The coefficient chart should be explained as follows:

- if a confidence interval crosses zero, that coefficient is not strongly distinguishable from zero at conventional significance levels,
- if a coefficient stays clearly away from zero, the corresponding predictor is more plausibly contributing to the conditional mean,
- the sign tells the direction of the estimated relationship,
- and the magnitude only makes sense relative to the standardized predictor scale.

In the current benchmark:

- `sp500_ret_l1` is clearly the dominant statistically significant predictor,
- `gold_ret_l1` is close to significance but not clearly beyond the threshold,
- most macro and FX effects are weak in this linear specification.

### How to interpret the residual visuals

The residual diagnostics should be read jointly, not one at a time.

#### Residual time-series plot

If the residuals were fully well-behaved, they would look like structureless noise around zero. Clustering, runs of same-sign residuals, or visibly persistent patterns indicate model misspecification.

#### Residual histogram

If the residual distribution were close to Gaussian, it would be fairly symmetric and not overly heavy in the tails. Financial residuals often show more extreme realizations than a normal distribution would predict.

#### Q-Q plot

If the Q-Q points follow the reference line closely, normality is plausible. If the tails bend away from the line, the residuals are heavy-tailed or skewed.

#### Ljung-Box bar chart

The 5% horizontal threshold is crucial:

- p-values above 0.05 mean the null of no residual autocorrelation is not rejected at that lag,
- p-values below 0.05 mean the model has likely left autocorrelation in the residuals.

### Example: how to explain the NASDAQ residual diagnostics

A rigorous, plain-language explanation is:

"The benchmark mean model is not completely adequate. At lag 5 the evidence of remaining autocorrelation is weak, but by lags 10 and 20 the Ljung-Box test strongly rejects white-noise residuals. The Q-Q plot and Jarque-Bera test also show that the residuals are not Gaussian and have heavier tails than a normal model would imply."

This means:

- the model captures part of the predictable mean structure,
- but not all of the serial dependence,
- and it definitely does not capture the heavy-tailed risk profile of the returns.

That is why the next phase should focus on volatility modeling rather than treating the residual variance as constant.

### Interpretation of the Phase 3 result

The key lesson is not that the model failed, but that it reached a meaningful limit:

- the mean process appears only weakly predictable with simple linear ARMA dynamics,
- the exogenous block explains some structure,
- but substantial residual dependence and heavy-tailed behavior remain.

This is exactly the type of result that motivates the next project stages:

- GARCH for conditional variance and volatility clustering,
- and richer forecasting models for nonlinear or cross-series structure.

## Phase 4 Implementation Update

Phase 4 has now been implemented in:

- `Assignment1/src/advml_assignment1/phase4_volatility_modeling.py`
- `Assignment1/scripts/run_phase4.py`

The saved outputs are stored in `Assignment1/outputs/phase4/`.

### Why GARCH is the correct next step

After Phase 3, the main unresolved issue was no longer the conditional mean alone. The residual diagnostics showed:

- remaining serial structure,
- strong non-normality,
- and most importantly, clear evidence of volatility clustering in squared residuals.

This is a standard empirical pattern in finance. Large shocks tend to be followed by large shocks, and calm periods tend to be followed by calm periods. The sign of the shock may change, but the **magnitude** tends to cluster.

That is exactly the setting in which GARCH models are useful.

### Intuition: mean forecasting versus volatility forecasting

It is crucial to separate two different forecasting tasks:

- **Mean forecasting**: predict the expected return
- **Volatility forecasting**: predict the expected uncertainty around that return

Phase 3 addressed the first task. Phase 4 addresses the second.

An analogy that works well in class is:

- Phase 3 asks, "Where is the center of the distribution?"
- Phase 4 asks, "How wide is the distribution likely to be today?"

### GARCH(1,1) built from first principles

#### Step 1 — The problem: variance is not constant

Classical regression and ARIMA models assume $\varepsilon_t \overset{\text{iid}}{\sim} \mathcal{N}(0, \sigma^2)$ — a fixed variance $\sigma^2$ that does not change over time. In financial markets this assumption is demonstrably wrong. Large moves cluster together (the 2020 COVID crash was followed by weeks of elevated daily swings, not isolated random spikes), and calm periods also cluster. This is **volatility clustering** — one of the strongest empirical regularities in finance.

If you look at $\varepsilon_t^2$ (squared residuals as a proxy for realized variance), you see clear positive autocorrelation across lags. A model that assumes constant variance is not capturing this structure.

#### Step 2 — ARCH: let variance depend on past squared shocks

**Engle (1982)** introduced the **AutoRegressive Conditional Heteroskedasticity (ARCH)** model. The key insight: model variance as a function of past squared shocks, not as a constant.

An **ARCH(q)** model:

$$
\sigma_t^2 = \omega + \alpha_1 \varepsilon_{t-1}^2 + \alpha_2 \varepsilon_{t-2}^2 + \cdots + \alpha_q \varepsilon_{t-q}^2
$$

Today's variance is a weighted sum of the last $q$ squared residuals plus a baseline $\omega$. A large shock at $t-1$ (large $\varepsilon_{t-1}^2$) raises today's estimated variance — the model responds to recent turbulence. The constraint $\omega > 0$ and $\alpha_i \geq 0$ ensures variance stays positive.

The problem with ARCH: to capture slowly decaying volatility, you need a very large $q$ (many lags), which introduces many parameters and instability.

#### Step 3 — GARCH: recycle the past variance estimate

**Bollerslev (1986)** extended ARCH with one additional term: the lagged variance estimate $\sigma_{t-1}^2$ itself. This is the **Generalized ARCH (GARCH)** model. A **GARCH(p, q)** model is:

$$
\sigma_t^2 = \omega + \underbrace{\alpha_1 \varepsilon_{t-1}^2 + \cdots + \alpha_q \varepsilon_{t-q}^2}_{\text{ARCH terms: } q \text{ lags of squared shocks}} + \underbrace{\beta_1 \sigma_{t-1}^2 + \cdots + \beta_p \sigma_{t-p}^2}_{\text{GARCH terms: } p \text{ lags of past variance}}
$$

The $\beta$ terms act as a long memory: instead of needing 20 ARCH lags to capture slow decay, one GARCH term $\beta_1 \sigma_{t-1}^2$ captures the accumulated history of past volatility efficiently.

#### Step 4 — GARCH(1,1): the standard specification

**GARCH(1,1)** uses $p=1$ and $q=1$ — one lag of each:

$$
\boxed{\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2}
$$

The (1,1) notation means: **1 ARCH lag** (one past squared shock) and **1 GARCH lag** (one past variance estimate). Empirically, GARCH(1,1) is sufficient for capturing volatility clustering in most financial return series — higher-order models rarely deliver material improvements at the cost of additional complexity.

**What each parameter does:**

| Parameter | Role | Intuition |
| --- | --- | --- |
| $\omega > 0$ | Baseline variance floor | Even in perfectly calm markets, variance cannot fall below $\omega$ |
| $\alpha \geq 0$ | ARCH coefficient — shock sensitivity | How strongly a large past shock ($\varepsilon_{t-1}^2$) immediately raises tomorrow's variance. High $\alpha$ = reactive. |
| $\beta \geq 0$ | GARCH coefficient — variance memory | How much of yesterday's variance estimate carries over. High $\beta$ = persistent, slow-decaying. |

**This project's fitted values:**

| Parameter | Estimated value | Interpretation |
| --- | --- | --- |
| $\omega$ | 0.02733 | Small baseline — most variance comes from the ARCH and GARCH terms, not the constant |
| $\alpha$ | 0.12767 | A large shock raises variance by ≈13% of that squared shock — moderate reactivity |
| $\beta$ | 0.86396 | 86% of yesterday's variance estimate carries into today — very strong memory |
| $\nu$ | 5.8907 | Student-t degrees of freedom — heavy tails (see below) |

#### Step 5 — The full model: variance + distribution

The complete GARCH(1,1) generative model for the residuals is:

$$
\varepsilon_t = \sigma_t \cdot z_t, \qquad z_t \overset{\text{iid}}{\sim} t_\nu(0, 1)
$$

$$
\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2
$$

At each time step: draw a standardized shock $z_t$ from a Student-t distribution with $\nu$ degrees of freedom, scale it by the current conditional standard deviation $\sigma_t$, and that gives the residual. The conditional variance recursion updates $\sigma_t^2$ based on the previous shock and the previous variance estimate.

#### Step 6 — Persistence and long-run variance

**Persistence** = $\alpha + \beta$. For this project: $0.12767 + 0.86396 = 0.9916$.

Persistence controls how fast a volatility shock decays. If today's variance is shocked above the long-run level, it decays back geometrically at rate $(\alpha + \beta)$ per period:

$$
\sigma_{t+k}^2 \to \bar{\sigma}^2 \text{ at rate } (\alpha + \beta)^k
$$

At 0.9916, this decay is extremely slow.

**Long-run (unconditional) variance** — the level variance reverts to:

$$
\bar{\sigma}^2 = \frac{\omega}{1 - \alpha - \beta} = \frac{0.02733}{1 - 0.9916} \approx \frac{0.02733}{0.0084} \approx 3.253
$$

(on the ×100 scaled residuals; converting back to log-return scale: $\sqrt{3.253} / 100 \approx 0.0181$, which matches the reported unconditional volatility of ≈ 1.81% per day)

**Half-life** — the number of periods for a volatility shock to decay to 50% of its initial deviation:

$$
\tau_{1/2} = \frac{\log(0.5)}{\log(\alpha + \beta)} = \frac{\log(0.5)}{\log(0.9916)} \approx \frac{-0.6931}{-0.00843} \approx 82.4 \text{ trading days}
$$

82 trading days ≈ 4 calendar months. A volatility shock (e.g., COVID crash, Fed rate hike) takes roughly four months to decay halfway back to the long-run level. This is why financial crises feel prolonged — the elevated uncertainty is genuinely persistent, not a statistical artifact.

### The GARCH(1,1) model

Let $\varepsilon_t$ be the residual from the Phase 3 mean model. A GARCH(1,1) model writes the conditional variance as

$$
\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2.
$$

Each term has a clear interpretation:

- $\omega$: the long-run variance floor
- $\alpha$: how strongly new shocks move current variance
- $\beta$: how persistent variance is over time

If $\alpha$ is large, volatility reacts strongly to new information. If $\beta$ is large, volatility decays slowly after a shock. The quantity

$$
\alpha + \beta
$$

is called **persistence**. When this sum is close to 1, volatility is highly persistent.

### Why the model uses zero mean

The GARCH model is fit on the residuals from the SARIMAX mean model. Since those residuals are already what remains after modeling the conditional mean, it is coherent to use a **zero-mean** volatility model in Phase 4.

In plain language:

- Phase 3 already tried to explain the average direction,
- so Phase 4 focuses only on the changing scale of uncertainty around zero.

### Why a Student-t distribution is used

The Phase 3 Jarque-Bera and Q-Q diagnostics already showed heavy tails. A Gaussian innovation assumption would therefore be too restrictive.

For this reason, the Phase 4 implementation uses a Student-t innovation distribution. This allows the model to accommodate heavier tails than a normal distribution.

The Student-t distribution has a degrees-of-freedom parameter, here denoted by $\nu$. Smaller $\nu$ means heavier tails. As $\nu \to \infty$, the Student-t approaches a Gaussian distribution.

### Why residuals are scaled before fitting

The model is fit on residuals multiplied by 100. This does **not** change the economics of the model; it is a numerical-stability step. Small floating-point magnitudes can make volatility optimization less stable, so percent-style scaling is standard practice.

After estimation, the saved variances and volatilities are converted back to the original return scale.

### How the out-of-sample volatility forecast is constructed

The Phase 4 evaluation uses:

- the Phase 3 training residuals for estimation,
- and the Phase 3 test forecast errors as the realized out-of-sample residual sequence.

The GARCH parameters are estimated on the training period only. Then one-step-ahead volatility forecasts are produced across the test period while the volatility recursion updates through time as new realized residuals become available.

This is the correct forecasting logic for a conditional variance model: we do not want to estimate volatility only in-sample and then pretend it generalizes automatically without any sequential updating.

### Additional risk metrics used in Phase 4

Because variance is harder to evaluate directly than the mean, the project uses:

- a realized squared error proxy, $\varepsilon_t^2$,
- forecast variance RMSE,
- forecast volatility RMSE,
- and QLIKE.

The QLIKE loss is

$$
QLIKE_t = \log(\hat{\sigma}_t^2) + \frac{\varepsilon_t^2}{\hat{\sigma}_t^2}.
$$

This loss is widely used in volatility forecasting because it is more robust than plain squared loss when realized variance is noisy.

### Empirical findings from Phase 4

The fitted model is:

- GARCH(1,1)
- innovation distribution: Student-t
- target series: Phase 3 NASDAQ mean residuals

Estimated parameters:

- $\omega \approx 0.02733$
- $\alpha_1 \approx 0.12767$
- $\beta_1 \approx 0.86396$
- $\nu \approx 5.8907$

Derived quantities:

- persistence: $\alpha_1 + \beta_1 \approx 0.9916$
- unconditional volatility: approximately `0.0181`
- volatility half-life: approximately `82.42` trading periods

### How to explain persistence and half-life

Persistence near 1 means volatility shocks fade slowly.

The half-life is the number of periods needed for a volatility shock to decay by half. A value around `82` trading days means that elevated volatility can remain influential for several months.

That is an economically plausible result for financial markets, especially after major macro or risk-off episodes.

### What the parameter estimates mean in plain language

The fitted coefficients say:

- volatility responds meaningfully to new shocks because $\alpha_1$ is clearly positive,
- volatility is highly persistent because $\beta_1$ is very large,
- and the tail behavior is materially heavier than Gaussian because $\nu$ is finite and relatively low.

This is exactly the type of pattern one expects from financial return residuals.

### Diagnostic improvement after GARCH filtering

This is the most important statistical takeaway from Phase 4.

After fitting the GARCH model, the standardized residual diagnostics show:

- Ljung-Box on standardized residuals is still significant at lags 5, 10, and 20,
- but Ljung-Box on **squared** standardized residuals is no longer significant,
- and the ARCH LM test also fails to reject remaining ARCH effects.

This means the model has substantially improved the **variance** dynamics even though it has not made the filtered residuals fully iid.

In other words:

- the volatility clustering is being captured much better,
- but some linear or distributional structure still remains in the standardized residuals.

### Why this is still a success

It is easy to think, "If the residuals are still not perfect, then the model failed." That would be the wrong interpretation.

A rigorous interpretation is:

- Phase 3 left strong dependence in squared residuals,
- Phase 4 removes most of that dependence,
- which means the model is doing its intended job well,
- even though it does not solve every remaining departure from ideality.

This is exactly how classical volatility modeling is supposed to be assessed.

### Example: how to explain the Phase 4 result orally

A strong short explanation is:

"The GARCH model does not try to predict whether the next residual is positive or negative. It tries to predict how volatile the next period is likely to be. The fitted model shows very persistent volatility, and after filtering, the squared standardized residuals no longer show strong autocorrelation. That means the volatility clustering has been modeled much better."

### Example: how to explain the volatility chart

In the visual notebook, the key volatility chart compares:

- forecast volatility,
- and a realized absolute-error proxy during the test period.

The correct way to explain that figure is:

"The forecast volatility series is smoother than the realized error path, because the model estimates latent conditional risk rather than raw noise. When realized shocks become larger, the forecast volatility tends to rise as the recursion updates, reflecting volatility clustering rather than isolated independent shocks."

### Example: how to explain the standardized residual diagnostics

A precise explanation is:

"Before GARCH, the squared residuals clearly clustered. After GARCH, the squared standardized residual tests are much cleaner, which shows that the conditional variance model has absorbed most of the volatility dynamics. However, the standardized residuals themselves still depart from iid behavior and still show heavy-tailed features."

### What the Student-t Q-Q plot means

Because the model is estimated with Student-t innovations, the Q-Q comparison is made against the fitted Student-t reference rather than a Gaussian reference. This is more rigorous than comparing against a normal distribution after explicitly fitting a heavy-tailed model.

If the points follow the fitted Student-t line more closely than the Phase 3 Gaussian Q-Q plot did, that is evidence that the heavy-tail specification is more realistic.

### Interpretation of the Phase 4 result

The core conclusion is:

- the GARCH model is highly persistent,
- it materially improves the variance dynamics,
- it removes most of the autocorrelation from squared standardized residuals,
- but it does not make the process perfectly iid.

This is exactly the result we wanted before moving to Phase 5. It gives us:

- a credible classical volatility benchmark,
- a risk forecast series,
- and a more complete decomposition of mean and variance behavior in the residual process.

## Phase 5 Implementation Update

Phase 5 has now been implemented in:

- `Assignment1/src/advml_assignment1/phase5_deep_forecasting.py`
- `Assignment1/scripts/run_phase5.py`

The saved outputs are stored in `Assignment1/outputs/phase5/`.

### Why the implemented Phase 5 model is PatchTST-style rather than Chronos-2

At the design stage, there were two realistic advanced-model directions:

- a pretrained foundation model such as Chronos-2,
- or a trainable neural forecasting architecture such as PatchTST.

Chronos-2 remains conceptually relevant and will still be referenced in the methodology discussion because it represents the foundation-model family. However, for the **implemented local benchmark**, a PatchTST-style model is the more rigorous first choice in this environment for three reasons:

1. **Reproducibility**: the model can be trained entirely from project data inside the repository without depending on external pretrained weights.
2. **Pedagogical transparency**: every stage of the model is inspectable, from feature construction to loss optimization.
3. **Controlled comparison**: a local train/test experiment against the Phase 3 classical benchmark is easier to interpret than a zero-shot foundation-model output whose internal pretraining distribution is external to the course project.

So the implemented deep-learning benchmark is a **PatchTST-style multivariate transformer**, while Chronos-2 is documented as a natural extension for later benchmarking if pretrained-model access is desired.

### What PatchTST is trying to solve

The classical SARIMAX model in Phase 3 is fundamentally linear. It can handle lag structure and exogenous regressors, but it cannot flexibly model:

- nonlinear interactions between features,
- complex local temporal motifs,
- or high-dimensional cross-series patterns that may matter only in certain market regimes.

PatchTST addresses these limits by using the transformer idea on **temporal patches** rather than raw point-by-point sequences.

### Phase 5 glossary for presentation

This subsection is written for explaining the project to technically literate people who may not come from finance.

#### Lookback window

The **lookback window** is how much past history the model is allowed to see before making one forecast.

In this project, the lookback window is `60` trading days. That means each forecast uses the previous 60 market observations as context.

A simple explanation is:

"The model is not looking at the entire history every time. It is looking at the most recent 60 trading days and trying to infer tomorrow's return from that recent context."

#### Patch length

The **patch length** is the number of consecutive days grouped together into one local segment before entering the transformer.

Here, the patch length is `10` trading days.

So each patch is a 10-day slice of recent market behavior.

#### Patch stride

The **patch stride** is how far we move the patching window each time we create the next patch.

Here, the stride is `5` days.

That means the patches overlap:

- patch 1 covers days 1 to 10,
- patch 2 covers days 6 to 15,
- patch 3 covers days 11 to 20,
- and so on.

This overlap is useful because financial patterns do not usually start and stop exactly on rigid boundaries.

#### Token

In transformer language, a **token** is the basic unit processed by attention.

In language models, a token might be a word fragment. In this project, a token is a **time patch**.

So instead of treating each day as a token, the model treats each 10-day segment as a token.

#### Channel

A **channel** is one feature stream observed through time.

Examples of channels in this project are:

- NASDAQ log return,
- S&P 500 RSI,
- gold MACD histogram,
- oil Bollinger z-score,
- GDP growth proxy.

The model sees many channels at once, which is why it is a **multivariate** forecaster.

### Intuition: why patching the time axis helps

Suppose we use the last 60 trading days to forecast tomorrow's NASDAQ return.

A naive transformer could treat each day as one token, which would produce 60 tokens. PatchTST instead groups nearby days into short local segments. In this project:

- lookback window = 60 days,
- patch length = 10 days,
- patch stride = 5 days.

This creates overlapping local summaries of the recent past. The resulting number of patches is

$$
N_{patch} = 1 + \frac{L - P}{S}
$$

where:

- $L = 60$ is the lookback window,
- $P = 10$ is the patch length,
- $S = 5$ is the stride.

So here:

$$
N_{patch} = 1 + \frac{60 - 10}{5} = 11.
$$

This means the transformer processes 11 learned temporal fragments rather than 60 single-day points.

The intuition is simple:

- a patch can represent a short market episode,
- such as a small drawdown, rebound, momentum burst, or sideways consolidation,
- and attention can then learn which of those historical fragments matter most for the next-day forecast.

### Mathematical structure of the implemented model

Let the multivariate input window be

$$
X_t \in \mathbb{R}^{L \times C},
$$

where:

- $L = 60$ is the lookback length,
- $C = 33$ is the number of input features.

Each channel is patched along the time axis into segments of length $P = 10$. Each patch is projected through a learned linear map into a latent representation of dimension $d_{model} = 32$.

If a patch vector is $x_{patch} \in \mathbb{R}^{10}$, the embedded token is

$$
z_{patch} = W x_{patch} + b,
$$

where $W \in \mathbb{R}^{32 \times 10}$.

Positional embeddings are added so the model can distinguish where each patch lies in the lookback window. The embedded patch sequence is then passed through a transformer encoder.

The transformer attention mechanism computes weights of the form

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

Intuitively:

- $Q$ asks what the current token is looking for,
- $K$ describes what each historical token contains,
- $V$ carries the information to be aggregated.

If two patches are relevant to one another, the attention score between them becomes large.

After encoding the patch sequence, the model pools the encoded patch information channel by channel, concatenates the channel summaries, and passes them through a small prediction head to produce a one-step-ahead return forecast.

### Why this still counts as a PatchTST-style model

The exact implementation is intentionally compact so it can train reproducibly on CPU. But it preserves the core PatchTST ideas:

- patching the time axis,
- shared patch embedding logic,
- transformer encoding over patch tokens,
- and forecasting from the encoded patch representation.

So this is not a generic feedforward network with a transformer label attached. It is genuinely a PatchTST-style design adapted to the scale of the assignment.

### Architecture provenance: paper versus implementation

This point is important academically: the Phase 5 architecture is **not invented ad hoc**. It is derived from a real research line, specifically the PatchTST model introduced by Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam in *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers* and released with an official implementation.

The original PatchTST paper emphasizes two core design ideas:

1. **Patching**: segment the time axis into subseries-level patches and use those patches as transformer tokens.
2. **Channel-independence**: treat each channel as a univariate series while sharing the same embedding and transformer weights across channels.

Those two ideas are explicitly stated in both the paper and the official repository README.

What our implementation keeps faithfully:

- patching the lookback window into overlapping temporal segments,
- linear patch embedding,
- positional embeddings,
- transformer encoding over patch tokens,
- channel-wise processing before final forecast aggregation.

What our implementation simplifies relative to the original research code:

- it is a compact one-step-ahead forecasting model rather than a full long-horizon benchmarking framework,
- it is designed for CPU reproducibility in a course environment,
- it uses a smaller custom head instead of reproducing the complete official experiment stack,
- and it uses our project-specific multivariate feature table rather than the standard benchmark datasets used in the paper.

So the correct characterization is:

- it is **inspired by and grounded in the PatchTST paper and official implementation**,
- but it is **not a line-by-line reproduction** of the official training code.

That distinction matters for rigor. We should not claim "this is exactly the published PatchTST benchmark." The honest and correct claim is:

"This is a PatchTST-style architectural adaptation built from the main ideas of the PatchTST paper and official implementation, scaled to the objectives and compute constraints of this assignment."

### Why a compact adaptation was the right engineering choice here

The original PatchTST paper is primarily aimed at long-term forecasting benchmarks and uses a broader experimental framework. Our assignment has a different objective:

- explainability,
- reproducibility,
- direct comparison against classical finance baselines,
- and local execution in the project environment.

Because of that, a compact adaptation is better than blindly copying the full research codebase.

It lets us:

- inspect every data transformation,
- align the target and features exactly to our financial setting,
- control leakage carefully,
- and explain the architecture in class without relying on a large external framework that hides important implementation details.

### Relation to the official channel-independence idea

The official PatchTST design is strongly centered on **channel-independence**, meaning each variable is processed as a univariate stream with shared weights across channels.

Our implementation keeps that spirit partially:

- patches are formed per channel,
- the same patch embedding layer is used across channels,
- the same transformer encoder weights are reused across channels after reshaping.

Then, unlike a purely channel-independent forecast head, our model concatenates the pooled channel representations before the final prediction layer. This is a pragmatic adaptation because our assignment is explicitly multivariate and we want the final head to combine information across assets, indicators, FX variables, and macro features for one NASDAQ forecast.

So again, the right statement is not "identical to official PatchTST," but rather:

"architecturally consistent with PatchTST's patching and shared-channel-processing principles, with a project-specific aggregation head for our supervised multivariate return task."

### Input features used in the deep benchmark

The deep model uses 33 lagged features:

- 7 asset log returns,
- 7 RSI(14) features,
- 7 MACD histogram features,
- 7 Bollinger z-score features,
- 2 lagged FX log returns (`eur_usd`, `usd_chf`),
- 3 lagged macro-change features (`gdp_growth_l1`, `cpi_inflation_l1`, `rate_change_l1`).

The target is the next-day `nasdaq log_return`.

All inputs are lagged by one day relative to the target. This is essential. It ensures the model does **not** use same-day information that would only be known after the target return has already occurred.

### Feature glossary for non-finance audiences

#### Log return

The log return between prices $P_{t-1}$ and $P_t$ is

$$
r_t = \log\left(\frac{P_t}{P_{t-1}}\right).
$$

It is a scale-consistent way to measure percentage-like changes through time. For small moves, it is close to the ordinary percentage return, but it behaves better mathematically in time-series modeling.

#### RSI(14)

RSI is a 14-period momentum oscillator on a 0 to 100 scale.

Intuition:

- high RSI means recent positive moves have dominated recent negative moves,
- low RSI means recent negative moves have dominated.

It is not a direct forecast by itself. In this project, it is one numerical feature among many.

#### MACD histogram

MACD compares a short-horizon exponential moving average with a longer-horizon one. The histogram is the difference between the MACD line and its signal line.

Intuition:

- positive histogram means short-term momentum is stronger than the smoothed trend signal,
- negative histogram means short-term momentum is weaker than that signal.

#### Bollinger z-score

The Bollinger z-score used in the code is

$$
z_t = \frac{P_t - \mu_t}{\sigma_t},
$$

where:

- $P_t$ is the current price,
- $\mu_t$ is the rolling 20-day mean,
- $\sigma_t$ is the rolling 20-day standard deviation.

This tells us how many rolling standard deviations the current price is above or below its local average.

Examples:

- $z_t = 0$: price is exactly at the local rolling mean,
- $z_t = 2$: price is about two local standard deviations above the mean,
- $z_t = -1.5$: price is about one and a half local standard deviations below the mean.

This is often easier to explain to technical audiences than the raw Bollinger bands because it is a standardized distance measure.

### Example of the leakage logic

Suppose the model predicts the NASDAQ return for trading day $t$.

It uses:

- the previous 60 days of lagged returns,
- the previous 60 days of lagged technical indicators,
- the previous 60 days of lagged macro-change and FX features.

It does **not** use day-$t$ technical indicators or day-$t$ close information when forecasting day $t$'s return.

This is the correct forecasting setup. Otherwise, the model would accidentally read information from the future.

### Train / validation / test split

The Phase 5 design matrix contains `3603` rows after dropping the rows needed for lagged features and transformed macro variables.

The split is:

- training rows through `2022-10-11`,
- validation rows from `2022-10-12` to `2023-10-16`,
- test rows from `2023-10-17` to `2024-10-18`.

At the window level, the model uses:

- `3039` training windows,
- `252` validation windows,
- `252` test windows.

This split is deliberately aligned with the earlier classical benchmark so the model comparison is fair.

### Why scaling is necessary

The input variables live on very different numerical scales:

- RSI is roughly between 0 and 100,
- log returns are small decimal numbers,
- MACD histogram values depend on asset price scale,
- macro changes are lower-frequency and numerically different again.

Without scaling, the optimization would be dominated by the largest-scale variables.

So the model standardizes:

- each input feature using the training-period mean and standard deviation,
- and the target return using the training-period mean and standard deviation.

This is standard practice in neural forecasting and avoids leaking test-set distributional information into training.

### Training setup

The implemented training configuration is:

- lookback window: `60`,
- patch length: `10`,
- stride: `5`,
- latent dimension: `32`,
- attention heads: `4`,
- transformer layers: `2`,
- feedforward dimension: `64`,
- dropout: `0.10`,
- optimizer: AdamW,
- learning rate: `0.001`,
- weight decay: `0.0001`,
- batch size: `64`,
- early stopping patience: `8`.

The model is trained on CPU and uses early stopping based on validation loss. This is important because daily return prediction is noisy and overfitting appears quickly.

### Optimization glossary

#### Epoch

An **epoch** means one complete pass through the training set.

If the training set has many windows, one epoch means the model has seen all of them once.

#### Batch size

The **batch size** is how many training windows are processed together before one optimizer update.

Here, the batch size is `64`.

#### Learning rate

The **learning rate** controls how large the parameter updates are during optimization.

If it is too large, training becomes unstable. If it is too small, training becomes very slow.

#### Weight decay

**Weight decay** is a regularization term that discourages unnecessarily large parameter values. It helps reduce overfitting.

#### Dropout

**Dropout** randomly hides part of the network during training. This forces the model not to rely too heavily on any single internal pathway and usually improves generalization.

#### Early stopping

**Early stopping** means training is halted once validation performance stops improving for a certain number of epochs.

This prevents the model from continuing to memorize the training set after the generalization benefit has already peaked.

#### Gradient clipping

**Gradient clipping** limits how large the gradient norm can become during backpropagation. This is a stability mechanism that reduces the risk of extremely large parameter updates.

### Why early stopping matters here

The training loss continues to fall after several epochs, but the validation loss stops improving earlier. That is exactly the pattern expected when a flexible model begins to fit noise rather than generalizable structure.

The best epoch in the saved run is epoch `9`.

This is a textbook case of why validation monitoring is necessary in time-series deep learning.

### Empirical results from Phase 5

The implemented Phase 5 model produces the following holdout metrics on the test period:

- RMSE: `0.010898`
- MAE: `0.008096`
- directional accuracy: `0.5714`
- mean forecast error: `0.000238`
- forecast/actual correlation: `0.1269`

For comparison, the Phase 3 SARIMAX benchmark produced:

- RMSE: `0.011026`
- MAE: `0.008240`
- directional accuracy: `0.5278`

So the deep model improves all three main holdout metrics, but only modestly.

### Evaluation metric glossary

#### RMSE

The **Root Mean Squared Error** is

$$
RMSE = \sqrt{\frac{1}{n}\sum_{t=1}^{n}(y_t - \hat{y}_t)^2 }.
$$

It penalizes large forecast errors more strongly than small ones because the errors are squared before averaging.

A lower RMSE is better.

#### MAE

The **Mean Absolute Error** is

$$
MAE = \frac{1}{n}\sum_{t=1}^{n}|y_t - \hat{y}_t|.
$$

It measures the average absolute forecast miss without squaring the errors.

A lower MAE is better.

#### Hit rate / directional accuracy

The **hit rate** checks whether the model got the sign of the return correct:

$$
HitRate = \frac{1}{n}\sum_{t=1}^{n}\mathbf{1}\{\text{sign}(y_t)=\text{sign}(\hat{y}_t)\}.
$$

If the actual return is positive and the forecast is positive, that counts as correct. If one is positive and the other negative, that counts as incorrect.

This is useful in finance because even if the exact magnitude is hard to predict, the sign can still matter economically.

#### Forecast/actual correlation

This is the correlation between the predicted returns and the realized returns over the evaluation period.

If it is positive, the forecasts and realizations tend to move in the same direction. If it is near zero, the forecasts contain little linear alignment with the realized outcomes.

### What the improvement means

This is the correct rigorous interpretation:

- the model is better than the classical benchmark,
- but the improvement is incremental rather than dramatic.

That is not disappointing. It is actually realistic.

Daily financial return forecasting is a low signal-to-noise problem. Even when a model is useful, the next-day expected return is usually small relative to the realized variability of returns. So one should **not** expect a deep model to produce large-amplitude deterministic forecasts.

### Why the forecast path still looks compressed around zero

In the saved Phase 5 predictions, the forecast series has much smaller variance than the realized return series.

This is normal in noisy return forecasting with mean-squared-error training:

- the model is penalized heavily for very large incorrect forecasts,
- so it tends to issue conservative estimates close to the conditional mean,
- especially when the data-generating process is noisy.

In plain language:

- the model does not "know" tomorrow's exact move,
- it learns small shifts in expected return conditional on recent history.

### How to explain the low-but-positive forecast correlation

The forecast/actual correlation is about `0.1269`.

That is not a large number, but in daily return prediction it is not meaningless either. A positive correlation means the forecasts and outcomes move together to some extent, but only weakly.

The correct explanation is:

"The model extracts some predictive structure, but the relationship is weak because next-day market returns are dominated by noise, new information arrival, and regime changes that no short-history model can explain fully."

### Why the hit rate improvement matters

The Phase 5 directional accuracy is about `57.1%`, compared with about `52.8%` for the classical baseline.

This is useful because in many financial decision settings, **getting the sign right more often** can matter even when the magnitude forecast is conservative.

However, this should still be presented carefully:

- hit rate alone is not enough,
- it can be regime-dependent,
- and it must eventually be tested under transaction costs and backtesting logic.

That is exactly why Phase 6 will focus on rolling evaluation and market-friction-aware backtesting rather than stopping at a single holdout error table.

### What the validation results tell us

The validation directional accuracy is only about `48.4%`, while the final test directional accuracy is higher.

This tells us something important: model performance is still regime-sensitive. The later holdout period is somewhat more favorable to the model than the validation period.

So the honest conclusion is not:

- "the model has solved return prediction,"

but rather:

- "the model shows some out-of-sample skill, but that skill is not uniformly stable across all subperiods."

This is exactly why rolling-window backtesting is necessary in the next phase.

### Example: how to explain PatchTST to a non-technical audience

A simple explanation is:

"Instead of looking at each day in isolation, the model groups recent days into short chunks and learns patterns across those chunks. It then uses attention to decide which recent chunks matter most for forecasting tomorrow's NASDAQ return."

### Example: how to explain the Phase 5 result orally

A strong classroom explanation is:

"The deep model slightly outperforms the classical SARIMAX benchmark on the final holdout period. That suggests there is some nonlinear or cross-series structure that the linear model misses. But the gain is moderate, not dramatic, and the forecasts remain close to zero. This is consistent with the idea that daily financial returns are highly noisy and only weakly predictable."

### Example: how to explain the training-curve figure

The correct explanation is:

"The training error keeps falling, but the validation error stops improving much earlier. That means the model can fit the training sample increasingly well, but additional fitting does not necessarily improve generalization. Early stopping is therefore essential."

### Example: how to explain the actual-vs-forecast scatter

The points do not lie tightly on the 45-degree line. That means the model does not explain the full amplitude of realized returns. But if there is some positive alignment rather than a completely structureless cloud, that supports the claim that the model captures a weak predictive signal.

### Code architecture walkthrough of `phase5_deep_forecasting.py`

This subsection explains what each main class and method is doing, so the implementation is not a black box.

#### `PhaseFiveConfig`

`PhaseFiveConfig` is the configuration container. It defines:

- where the Phase 1 modeling data lives,
- where Phase 3 benchmark metadata lives,
- where Phase 5 outputs will be saved,
- the target column,
- the split sizes,
- the model hyperparameters,
- and the training hyperparameters.

This is useful because the experiment settings are centralized rather than scattered across the code.

#### `PhaseFiveArtifacts`

`PhaseFiveArtifacts` is the output container returned by the pipeline. It holds:

- the final design matrix,
- the training history,
- validation predictions,
- test predictions,
- the feature schema,
- the metadata summary,
- and the saved model state dictionary.

This makes the pipeline easy to inspect and serialize.

#### `PatchTSTForecaster` — full tensor shape trace

This is the neural network itself. The config values that drive all shapes are: `lookback_window=60`, `patch_length=10`, `patch_stride=5`, `num_channels=33`, `d_model=32`, `num_heads=4`, `num_layers=2`, `feedforward_dim=64`. Let B denote batch size.

**Constructor — what each layer is:**

- `patch_embedding = nn.Linear(10, 32)`: projects each 10-day raw patch into a 32-dim token. One shared linear layer used across all channels and all patch positions.
- `position_embedding = nn.Parameter(zeros(1, 1, 11, 32))`: learnable positional offset, one vector per of the 11 patch slots. Initialized near zero, learned during training. Tells the model which patch (early, middle, late in the 60-day window) each token represents.
- `encoder = nn.TransformerEncoder(layer, num_layers=2)`: stack of 2 Pre-LN Transformer encoder layers, each containing Multi-Head Self-Attention (4 heads, d_model=32) followed by a FeedForward block (32 → 64 → 32, GELU activation).
- `channel_norm = nn.LayerNorm(32)`: normalizes the pooled per-channel representation before the head.
- `head = nn.Sequential(Flatten, Linear(1056→64), GELU, Dropout(0.10), Linear(64→1))`: two-layer MLP that mixes all 33 channels and outputs one scalar.

**`num_patches` formula** (computed at construction, line 101):

$$N_{\text{patches}} = 1 + \frac{L - P}{S} = 1 + \frac{60 - 10}{5} = 11$$

where $L=60$ is the lookback window, $P=10$ is the patch length, and $S=5$ is the stride.

**Forward pass — step by step with tensor shapes:**

```text
Input: [B, 60, 33]              (batch, lookback days, feature channels)
```

**Step 1 — Transpose** (line 129):

```python
series = inputs.transpose(1, 2)
# [B, 60, 33] → [B, 33, 60]
```

Move the channel dimension before the time dimension so `unfold` can slide along time per channel.

**Step 2 — Patch extraction with unfold** (line 130):

```python
patches = series.unfold(dimension=2, size=10, step=5)
# [B, 33, 60] → [B, 33, 11, 10]
```

`unfold` slides a window of size 10 with step 5 along dimension 2 (the time axis). Produces 11 overlapping patches, each 10 days long. The patches overlap by 5 days — patch 0 covers days 0–9, patch 1 covers days 5–14, ..., patch 10 covers days 50–59.

**Step 3 — Patch projection + positional embedding** (line 131):

```python
tokens = self.patch_embedding(patches) + self.position_embedding
# Linear(10→32) applied to last dim: [B, 33, 11, 10] → [B, 33, 11, 32]
# position_embedding shape: [1, 1, 11, 32] → broadcasts to [B, 33, 11, 32]
# result: [B, 33, 11, 32]
```

Each 10-day patch becomes a 32-dim token. The position embedding adds a learned offset that encodes *where* in the window the patch sits.

**Step 4 — Channel-independent Transformer encoding** (lines 133–134):

```python
encoded = self.encoder(tokens.reshape(B * 33, 11, 32))
# reshape: [B, 33, 11, 32] → [B*33, 11, 32]
# encoder processes B*33 independent sequences of 11 tokens
# output: [B*33, 11, 32]
```

This is the **key trick for channel-independence**: merging the batch and channel dimensions into one "super-batch" forces the Transformer to treat each channel's patch sequence as completely independent. The encoder weights are *shared* across all 33 channels (same parameters used for each channel), but no channel attends to another — self-attention is only within a channel's own 11 tokens.

Inside each of the 2 encoder layers (Pre-LN = LayerNorm first, then attention/FFN):

1. LayerNorm → Multi-Head Self-Attention (4 heads, keys/queries/values all in ℝ^8 per head) → residual add
2. LayerNorm → FeedForward (32 → 64 → 32, GELU) → residual add

**Step 5 — Mean pool over patches + channel reassembly** (line 135):

```python
pooled = self.channel_norm(encoded.mean(dim=1)).reshape(B, 33, 32)
# encoded: [B*33, 11, 32]
# .mean(dim=1): average over 11 patches → [B*33, 32]
# channel_norm: LayerNorm(32)
# .reshape(B, 33, 32): reassemble channels
```

Instead of flattening all 11 patch representations per channel (which would give 11×32=352 dims per channel, 11,616 total — too large for the training set), we average-pool across the patch dimension. This compresses each channel's temporal information into a single 32-dim summary vector.

**Step 6 — Cross-channel prediction head** (lines 117–136):

```python
forecast = self.head(pooled).squeeze(-1)
# Flatten: [B, 33, 32] → [B, 1056]
# Linear(1056→64): cross-channel mixing
# GELU + Dropout(0.10)
# Linear(64→1): scalar forecast
# squeeze: [B, 1] → [B]
```

The head explicitly mixes information from all 33 channels. This is where the model learns which combination of asset returns, RSI signals, MACD, Bollinger bands, FX moves, and macro features is most predictive of tomorrow's NASDAQ return.

**Complete shape flow summary:**

| Stage | Tensor shape | Operation |
| --- | --- | --- |
| Input | [B, 60, 33] | Raw lookback window |
| After transpose | [B, 33, 60] | Channel-first |
| After unfold | [B, 33, 11, 10] | 11 overlapping 10-day patches |
| After patch embedding + pos | [B, 33, 11, 32] | Tokens in d_model space |
| After reshape for encoder | [B×33, 11, 32] | Channel-independent view |
| After Transformer encoder | [B×33, 11, 32] | Context-enriched tokens |
| After mean pool + reshape | [B, 33, 32] | One vector per channel |
| After flatten | [B, 1056] | All channels concatenated |
| After MLP head | [B, 1] | Next-day return forecast |

### Paper vs implementation: a direct comparison

| Aspect | Original PatchTST paper | Our implementation | Reason for deviation |
| --- | --- | --- | --- |
| **Patching** | Non-overlapping (stride = patch_length) | Overlapping (stride=5 < patch_length=10) | More tokens from short 60-day window (11 vs 6) |
| **Channel independence** | Strict — separate output per channel, no cross-channel mixing anywhere | Partial — encoder is channel-independent, but head mixes all channels | We need a single NASDAQ forecast from all 33 channels |
| **Instance normalization** | RevIN — reversible per-instance normalization inside the model | External z-score standardization on training stats | Simpler, sufficient for near-zero-mean return series |
| **Patch pooling** | Flatten all patch representations → large linear head | Mean pool over patches → compact head | Avoids 11,616-dim head that would overfit on ~3000 windows |
| **Output** | Multi-step forecast: T future steps (24, 96, 192, 720 in paper) | One-step-ahead scalar (next-day return) | Assignment objective is next-day return, not long-horizon |
| **Encoder normalization** | Post-LN (original) | Pre-LN (`norm_first=True`) | Pre-LN is more stable for small models and short training |
| **Activation** | ReLU in FFN | GELU | Smoother gradient landscape; standard in modern transformers |

#### `PatchTSTDeepForecaster.run`

This is the orchestration method. It:

1. fixes the random seed,
2. loads the Phase 1 modeling dataset,
3. builds the lagged design matrix,
4. scales the features and target,
5. creates rolling windows,
6. instantiates the neural model,
7. trains it,
8. reloads the best validation checkpoint,
9. makes validation and test forecasts,
10. builds the metadata and comparison summary.

#### `_build_design_matrix`

This method constructs the modeling table for Phase 5.

It selects the target and then creates lagged versions of:

- asset returns,
- RSI features,
- MACD histogram features,
- Bollinger z-scores,
- FX returns,
- and macro changes.

It also builds `feature_schema`, which is a descriptive table explaining what each feature is and what category it belongs to.

#### `_prepare_scaled_arrays`

This method:

- separates inputs from target,
- defines the train, validation, and test boundaries,
- computes scaling parameters on the training set only,
- standardizes features and target,
- and returns the arrays needed by the downstream window builder.

The reason scaling is done here is to ensure that later code works only with normalized arrays, while the saved metadata preserves the original scaling values.

#### `_build_window_datasets` and `_build_windows`

These methods create the actual supervised learning examples.

For each target day $t$, the model input is the block of features from days $t-60$ through $t-1$, and the label is the target return at day $t$.

This is the critical transformation from a time-indexed table into a neural forecasting dataset.

#### `_fit_model`

This method trains the neural network.

It creates data loaders, defines:

- the AdamW optimizer,
- mean squared error loss,
- gradient clipping,
- and early stopping.

For each epoch it:

- loops over training batches,
- computes predictions,
- computes loss,
- backpropagates gradients,
- updates parameters,
- then evaluates the current model on the validation set.

The best model state is stored whenever validation loss improves.

#### `_predict_split`

This method runs the trained model in evaluation mode on one split and returns a dataframe with:

- dates,
- scaled targets,
- scaled predictions,
- and later, after inverse scaling, actual returns, forecasts, forecast errors, and directional-correctness flags.

#### `_build_metadata`

This method inverse-transforms the predictions back to the original return scale and computes:

- RMSE,
- MAE,
- hit rate,
- mean forecast error,
- forecast/actual correlation,
- split boundaries,
- feature counts,
- and comparison metrics versus the Phase 3 benchmark.

This is the central summary builder for the saved experiment record.

#### `PhaseFivePipeline`

This is the persistence layer. It runs the forecaster and writes:

- design data,
- training history,
- validation predictions,
- test predictions,
- feature schema,
- model metadata,
- and the saved PyTorch state dictionary

to `Assignment1/outputs/phase5/`.

### Interpretation of the Phase 5 result

The Phase 5 conclusion should be stated carefully:

- the PatchTST-style deep model outperforms the classical linear baseline on the fixed holdout sample,
- the improvement is real but modest,
- the forecast distribution remains conservative and centered near small expected returns,
- and the evidence strongly suggests that final evaluation must rely on rolling backtests rather than a single split.

That is exactly the right setup for Phase 6.

## Phase 6 Implementation Update

Phase 6 has now been implemented in:

- `Assignment1/src/advml_assignment1/phase6_backtesting.py`
- `Assignment1/scripts/run_phase6.py`

The saved outputs are stored in `Assignment1/outputs/phase6/`.

### What Phase 6 is for

This is an important conceptual point.

By the end of Phase 5, we already had:

- a classical forecasting benchmark,
- a volatility benchmark,
- a deep-learning benchmark,
- and a fixed train/test comparison.

But that still does **not** answer the full financial question.

A single holdout split tells us:

- how the models behaved on one particular historical segment.

It does **not** tell us:

- whether that performance is stable across different market regimes,
- whether the signal survives repeated retraining through time,
- or whether the forecast skill is economically useful once trading frictions are included.

That is why Phase 6 exists.

### The difference between forecast evaluation and backtesting

Forecast evaluation asks:

- "How close are the predictions to the realized values?"

Backtesting asks:

- "If I had repeatedly used these predictions historically to take positions, what would the realized trading behavior have looked like?"

Those are related questions, but they are not the same.

This project now demonstrates that difference directly.

### Rolling-window cross-validation logic

Phase 6 uses a rolling-window design with:

- training window = `2000` observations,
- validation window = `252` observations,
- test window = `252` observations,
- step size = `252` observations,
- total number of folds = `5`.

So the procedure is:

1. fit the model on the first rolling training block,
2. use the validation block for Phase 5 early stopping,
3. evaluate on the next unseen test block,
4. shift the window forward,
5. repeat the entire process.

This is much more defensible than a single split because it evaluates the models under multiple historical regimes.

### Is Phase 6 a backtest? Yes — and here is exactly what that means

Phase 6 is a **walk-forward backtest**. "Backtest" means: simulate what would have happened if you had used this model to trade in the past, with realistic constraints, starting from historical data. "Walk-forward" means the simulation respects time — you never use future data to make a past decision.

The question Phase 6 answers is not "how accurate are the forecasts?" — that was answered in Phases 3 and 5. The question is:

> "If I had repeatedly used this model to make trading decisions through different market regimes, with real transaction costs, what would have actually happened to a portfolio following this strategy?"

This is a much harder and more realistic question. A model can look excellent on RMSE tables and fail commercially, or look modest statistically and still generate consistent risk-adjusted returns. Phase 6 forces both models through that test.

### Concrete fold structure — what "rolling" means

The 5 folds step forward one year at a time, each requiring a full model refit:

```text
Fold 1: |----TRAIN (2000 days)----|--VAL (252)--|--TEST (252)--|
Fold 2:      |----TRAIN (2000 days)----|--VAL (252)--|--TEST (252)--|
Fold 3:           |----TRAIN (2000 days)----|--VAL (252)--|--TEST (252)--|
Fold 4:                |----TRAIN (2000 days)----|--VAL (252)--|--TEST (252)--|
Fold 5:                     |----TRAIN (2000 days)----|--VAL (252)--|--TEST (252)--|
                                                                  ↑
                                                     5 non-overlapping test windows
```

Each fold: re-estimate the SARIMAX order from scratch (auto_arima may select a different order than (0,0,0) in some folds), retrain the PatchTST model from random initialization, generate forecasts for the 252-day test window, apply the trading rule and costs, record the P&L. The 5 test windows are non-overlapping and cover a combined ~1,260 trading days of out-of-sample history across different market regimes.

### Why rolling retraining matters

Financial relationships drift over time.

Examples:

- macro sensitivity changes,
- cross-asset relationships change,
- volatility regimes change,
- trend and reversal behavior change.

If a model is only trained once and tested once, it may accidentally benefit from one favorable regime. Rolling retraining forces the model to adapt repeatedly and therefore gives a much better picture of temporal robustness.

### Trading rule used in the backtest

The strategy rule is intentionally simple so the model comparison stays interpretable.

For each forecasted day:

$$
position_t = \text{sign}(\hat{r}_t)
$$

where:

- $+1$ means take a long position,
- $-1$ means take a short position,
- $0$ means no directional conviction if the forecast is exactly zero.

The gross strategy return is

$$
R^{gross}_t = position_t \cdot r_t,
$$

where $r_t$ is the realized NASDAQ log return.

### Market frictions: commissions and slippage

The backtest includes:

- commission = `2` basis points,
- slippage = `3` basis points.

So the total trading cost rate is `5` basis points per unit turnover.

Turnover is defined as

$$
turnover_t = |position_t - position_{t-1}|.
$$

This is important because flipping from long to short is more expensive than staying long.

The transaction cost is

$$
cost_t = turnover_t \cdot c,
$$

where $c = 0.0005$.

The net strategy return is therefore

$$
R^{net}_t = R^{gross}_t - cost_t.
$$

### Why transaction costs matter

A model that trades too often can look good statistically but poor economically. If every small forecast change forces position changes, the apparent predictive edge can be consumed by costs.

That is why Phase 6 reports turnover explicitly.

### Financial KPI glossary

#### Net cumulative return

If the strategy net return on day $t$ is $R^{net}_t$, cumulative wealth is

$$
W_t = \prod_{i=1}^{t}(1 + R^{net}_i).
$$

The net cumulative return is $W_T - 1$.

#### Sharpe ratio

The annualized Sharpe ratio used here is

$$
Sharpe = \sqrt{252}\frac{\bar{R}^{net}}{\sigma(R^{net})}.
$$

It measures return per unit of realized variability. A higher Sharpe ratio is better.

#### Maximum drawdown

Maximum drawdown measures the worst peak-to-trough loss in the cumulative wealth path:

$$
Drawdown_t = \frac{W_t}{\max_{s \le t} W_s} - 1.
$$

The maximum drawdown is the minimum of this series.

This is one of the most intuitive risk metrics in finance because it answers:

"What was the worst percentage loss from a previous peak?"

#### Average turnover

Average turnover measures how often the strategy changes position.

Higher turnover generally means:

- higher cost drag,
- more sensitivity to noise,
- and more dependence on execution quality.

### Models evaluated in the rolling backtest

Phase 6 compares two forecasting engines:

- the Phase 3 SARIMAX benchmark,
- the Phase 5 PatchTST-style transformer.

The GARCH model from Phase 4 is not used here as a separate directional strategy because its primary role is volatility forecasting rather than sign forecasting. Its output remains useful for future risk-scaling extensions, but the current Phase 6 comparison focuses on the mean-forecasting models.

### Important design choice for the rolling PatchTST retraining

Repeated deep-model retraining inside each fold is computationally much heavier than re-fitting SARIMAX. For that reason, the rolling backtest uses a **lighter retraining budget** than the full Phase 5 single-split experiment:

- maximum epochs per fold = `10`,
- early stopping patience = `4`.

This is not a shortcut that changes the model family. It is a pragmatic engineering choice so the rolling backtest remains reproducible locally while still re-estimating the deep model on every fold.

### Empirical results from Phase 6

Overall forecast metrics across the rolling out-of-sample predictions:

#### SARIMAX

- RMSE: `0.016122`
- MAE: `0.011522`
- directional accuracy: `0.5183`

#### PatchTST-style model

- RMSE: `0.015967`
- MAE: `0.011231`
- directional accuracy: `0.5349`

So the deep model remains slightly better on the prediction metrics.

### Trading results after costs

#### SARIMAX

- average turnover: `0.6865`
- gross cumulative return: `1.5564`
- net cumulative return: `0.6581`
- annualized Sharpe ratio: `0.5251`
- maximum drawdown: `-0.4042`

#### PatchTST-style model

- average turnover: `0.2262`
- gross cumulative return: `0.8394`
- net cumulative return: `0.5947`
- annualized Sharpe ratio: `0.4955`
- maximum drawdown: `-0.4259`

### The key financial lesson from Phase 6

This is the most important practical conclusion of the project so far:

- the deep model is still slightly better statistically,
- but it is **not** clearly better economically in the rolling backtest.

In fact:

- PatchTST improves RMSE, MAE, and hit rate,
- but SARIMAX achieves a slightly higher Sharpe ratio,
- a slightly higher net cumulative return,
- and a slightly smaller maximum drawdown.

That is a highly realistic financial machine learning result.

### Why this can happen — the four mechanisms in depth

This is the conceptually deepest part of Phase 6. Understanding it fully requires decomposing exactly why a statistically superior model can be economically inferior.

#### Mechanism 1 — Hit rate is uniformly weighted; P&L is not

Directional accuracy counts a correctly-predicted 0.01% return day identically to a correctly-predicted 2.0% return day. Both are scored as "1 correct direction." But from a P&L perspective, getting the sign right on a ±2% day earns 40× more than getting the sign right on a ±0.05% day. RMSE also treats large errors more heavily than small ones (quadratic penalty), but it is still symmetric around zero — it does not distinguish whether the large error was on an important or unimportant day.

PatchTST achieves higher directional accuracy (53.5% vs 51.8%), but if its extra correctly-predicted days are concentrated on small-return days while it still misses the big moves, the gross P&L advantage is minimal. The SARIMAX strategy's lower accuracy may be distributed differently: it might miss more small-return days but catch more of the large return days, which dominate cumulative wealth.

#### Mechanism 2 — Forecast amplitude and position sizing

The trading rule uses `position_t = sign(forecast_t)` — the sign only, not the magnitude. But the magnitude of the forecast implicitly affects which days the model changes its sign. PatchTST's forecasts are more compressed toward zero (this is MSE-optimal behavior, as discussed in Phase 5). A near-zero forecast is much more likely to flip sign from one day to the next due to small random fluctuations than a model that produces larger-amplitude forecasts with more stable sign. This means:

- SARIMAX produces more decisive signals (larger magnitudes), which are more stable in sign → lower spurious flip rate on unimportant days → fewer wasted trades
- PatchTST's near-zero forecasts are sign-unstable → may stay out of the market on days that matter → lower turnover, but also missing profitable directional moves

The compression of PatchTST's forecasts is statistically correct (it minimizes MSE) but economically suboptimal (the strategy needs clear directional signals, not hedged near-zero predictions).

#### Mechanism 3 — Turnover asymmetry

SARIMAX has turnover 0.687 (changes position ~69% of days) vs PatchTST at 0.226 (changes position ~23% of days). This seems like a massive cost disadvantage for SARIMAX: at 5 bps per flip, SARIMAX pays ≈ 0.687 × 5 bps = 3.4 bps/day in cost drag, while PatchTST pays only ≈ 0.226 × 5 bps = 1.1 bps/day.

Yet SARIMAX still generates higher net returns. This means SARIMAX's higher-cost signals are *worth paying for* — the incremental gross P&L from acting on those signals exceeds the extra cost. PatchTST's lower-cost but more passive position changes do not generate enough gross P&L to overcome even the smaller cost it does pay. The economic value of the signal matters as much as its frequency.

#### Mechanism 4 — Regime dependence

The fold-by-fold directional accuracy chart shows that neither model consistently dominates across all 5 folds. In some market regimes (trending, mean-reverting, vol-spike), SARIMAX's linear lagged structure is better calibrated; in others, PatchTST's nonlinear feature interactions may be more useful. The aggregate winner is determined by which model happened to be better in the regimes that contribute most to the overall P&L — and this is not predictable in advance from the model's structural properties alone.

This is why: **the result from Phase 6 is a historical fact about this specific dataset and sample period, not a universal law.** On a different sample period, PatchTST might win economically. The finding is that the statistical ranking does not automatically determine the economic ranking.

### The correct evaluation chain for financial ML

Most ML papers stop at step 2 or 3. Phase 6 goes all the way to step 5:

| Step | Question | Metric | Phase |
| --- | --- | --- | --- |
| 1 | Does the model fit the training data? | In-sample RMSE, AIC | Phase 3, Phase 5 |
| 2 | Does the model generalize to new data? | Out-of-sample RMSE, MAE | Phase 3, Phase 5 |
| 3 | Does the model predict direction correctly? | Directional accuracy (hit rate) | Phase 3, Phase 5, Phase 6 |
| 4 | Does the directional skill generate gross P&L? | Gross cumulative return | Phase 6 |
| 5 | Does the strategy survive costs and risk? | Net return, Sharpe, max drawdown | Phase 6 |

The fact that PatchTST wins steps 1–3 but SARIMAX wins steps 4–5 is precisely what makes Phase 6 a meaningful and honest contribution to the project. Had Phase 6 simply confirmed "PatchTST wins economically too," the result would be consistent but less informative. The discrepancy between statistical and economic rankings is the insight.

### What the results do and do not prove

**What Phase 6 proves:**

- Over this specific historical sample with this specific trading rule and these specific cost assumptions, SARIMAX generates a slightly higher Sharpe ratio and net return than PatchTST
- Neither model achieves consistently high directional accuracy across all market regimes (fold-by-fold variability is large)
- The 40% maximum drawdown for both models is severe — a risk manager would likely impose position limits or stop-loss rules that this simple backtest does not include

**What Phase 6 does not prove:**

- That SARIMAX is universally better than PatchTST for financial forecasting
- That the statistical advantage of PatchTST is meaningless (it remains real across all holdout metrics)
- That either model would perform this well going forward (all backtests suffer from some degree of in-sample selection bias in the modeling choices, even with rolling windows)
- That the strategy is investable as-is (no slippage for large orders, no short-selling constraints, no risk limits are modeled)

### Why the lower-turnover deep model still does not dominate

One especially instructive result is this:

- PatchTST has much lower average turnover than SARIMAX,
- yet it still does not beat SARIMAX on Sharpe or net cumulative return.

This means the issue is **not only** trading costs. The deep model's signal is also economically weaker in terms of realized payoff, even though its statistical error metrics are slightly better.

That is an excellent teaching example because it shows:

- statistical edge and economic edge are related,
- but they are not identical.

### Fold-level behavior

The fold-by-fold results also matter.

Across the five folds:

- the SARIMAX order selected by `auto_arima` stays very simple, mostly `(0,0,0)` and once `(1,0,0)`,
- the deep model does not dominate every fold,
- and directional accuracy changes materially across folds.

This supports the conclusion that the forecasting problem is regime-dependent.

### Example: how to explain Phase 6 orally

A strong classroom explanation is:

"The deep model looked slightly better on a single holdout split, but rolling backtesting shows that this does not automatically translate into better trading performance. Once we retrain through time and include transaction costs, the classical SARIMAX strategy is still competitive and even slightly better on Sharpe and net return. This is why financial ML must evaluate economic utility, not only forecast error."

### Example: how to explain the cumulative wealth chart

The correct explanation is:

"This chart shows what would have happened to one unit of capital if we had followed the sign of each model's forecast through time and paid costs whenever the position changed. The wealth paths summarize not only prediction accuracy, but also timing, turnover, and the economic quality of the signal."

### Interpretation of the Phase 6 result

The final Phase 6 interpretation should be:

- the Phase 5 deep model remains statistically competitive,
- but the rolling financial evaluation is more ambiguous,
- and the classical benchmark remains economically relevant.

This is a strong result, not a disappointing one. It shows the project is being evaluated with the right level of rigor.

It also sets up the final report discussion well:

- deep learning added predictive flexibility,
- classical models remained hard to beat economically,
- and robust financial evaluation required rolling backtesting under frictions rather than a single favorable holdout.

## Hybrid LSTM and Chronos Foundation Model Benchmark

This benchmark is implemented in `Assignment1/03_Forecasting_LSTM_&_Chronos.ipynb`. It adds a further modeling layer on top of the structured six-phase pipeline, combining a Hybrid STL-LSTM architecture with the Amazon Chronos-T5 foundation model for zero-shot validation.

### Design Philosophy: Targeting the STL Residual

The central architectural choice in this notebook is that the LSTM targets not the raw return series, but the **STL residual component** of the selected asset's log price.

The motivation comes directly from Phase 2. STL decomposition separated each log-price series into:

$$\log P_t = T_t + S_t + R_t$$

The trend $T_t$ is smooth and deterministic — a statistical method can handle it. The seasonal $S_t$ is weak in most assets and adds little signal. The hard forecasting problem is the residual $R_t$, which captures idiosyncratic shocks, market dislocations, and nonlinear regime behavior.

The LSTM focuses its entire learning capacity on $R_t$. The final reconstructed price is then:

$$\hat{\log P}_t = T_t + S_t + \hat{R}_t^{LSTM}$$

where the trend and seasonal components come from the Phase 2 STL fit and the LSTM predicts only the stochastic shock. This is a principled decomposition strategy: it separates the "easy" structural part of the problem (handled classically) from the "hard" stochastic part (handled by deep learning).

### Important Note: Target Asset

The notebook targets the STL residual of the **S&P 500** series. This is a different target from the NASDAQ log return used in Phases 3, 5, and 6. The reason is that the notebook picks the first available asset from the Phase 2 STL output. This creates a meaningful conceptual difference:

- Phases 3, 5, 6: target is `nasdaq log_return` (return-space, no decomposition)
- Hybrid LSTM notebook: target is `sp500 STL residual` (decomposed log-price space)

Any metric comparison across these experiments must account for this difference. The results are not directly numerically comparable, but the modeling philosophy is complementary.

### Feature Engineering: Full Pipeline Integration

The LSTM integrates three sources of information from the structured pipeline:

1. **Phase 1 outputs** (`cleaned_data.csv`): all OHLCV columns and macro/FX variables across all assets
2. **Phase 2 outputs** (`stl_decomposition_components.csv`): STL trend, seasonal, and residual for the target asset
3. **Phase 4 outputs** (`test_volatility_forecasts.csv`): GARCH-based conditional variance and volatility forecasts, QLIKE contributions, realized squared errors

By including the GARCH forecast as a feature, the LSTM can observe the current estimated risk regime and condition its residual prediction accordingly. This is a sophisticated design: the deep model sees not just raw market data but also a classical model's assessment of current volatility.

Missing values are handled with forward-fill then backward-fill, which is more aggressive than the Phase 1 macro approach. This is acceptable because LSTM training is tolerant of moderate approximation errors in high-dimensional auxiliary inputs.

### LSTM Architecture

```text
FinancialLSTM:
  Input:   (batch, seq_length=30, n_features)
  LSTM:    2 stacked layers, hidden_size=64, dropout=0.1
  Linear:  64 → 1
```

Key design decisions and justifications:

| Choice | Value | Reason |
| --- | --- | --- |
| Stacked LSTM layers | 2 | Allows higher-order temporal dependencies |
| Hidden size | 64 | Sufficient capacity for ~55 features without excessive parameters |
| Dropout | 0.1 | Light regularization to reduce overfitting on noisy residuals |
| Sequence length | 30 | 1.5 trading months of lookback context |
| Gradient clipping | max_norm=1.0 | Prevents exploding gradients on noisy financial data |
| Adam lr | 0.0005 | Conservative to avoid overshooting on a difficult stochastic target |

The forward pass reads the last hidden state from the final LSTM layer and maps it to a scalar residual prediction. Both inputs and target are independently standardized with separate `StandardScaler` instances. This double-scaling prevents macro variables (GDP in trillions) from dominating the loss gradient relative to return-scale features.

### Training Behavior

Training runs for 50 epochs with the following loss trajectory:

| Epoch | Training Loss |
| --- | --- |
| 10 | 0.9607 |
| 20 | 0.9370 |
| 30 | 0.9060 |
| 40 | 0.8597 |
| 50 | 0.7921 |

The loss decreases monotonically across all 50 epochs without a divergence episode, which confirms that gradient clipping is working correctly. The continued decline at epoch 50 suggests the model could benefit from more training epochs, though early stopping based on a held-out validation set was not implemented in this notebook.

### Data Split

The split is 80% training / 20% test applied sequentially:

- Training windows: approximately 80% of the windowed sequence pool
- Test samples: 728 windows

This is different from the phase-based splits in Phases 3, 5, and 6, which use a fixed 252-day test window ending at 2024-10-18. The 728-sample test set is substantially larger and covers multiple market regimes.

### Amazon Chronos Zero-Shot Forecast

Chronos is a language-model-style foundation model for time series, pretrained on a large and diverse collection of time-series data. The key property of Chronos is that it generates forecasts without any task-specific training: it is a **zero-shot** benchmark.

The model used here is `chronos-t5-base`. Key parameters:

- **Context**: the full historical log-price series from the dataset (not just returns)
- **Prediction length**: 5 trading days
- **Output**: 20 sample trajectories from which quantiles are extracted
- **Quantiles used**: Q10, Q50 (median), Q90 → forming an 80% prediction interval

The median trajectory is used as the point forecast. The prediction interval width (Q90 - Q10) serves as a volatility proxy.

A critical practical note: Chronos requires downloading pretrained model weights on first run. The `chronos` package from Amazon must be installed. The model is loaded with `device_map="cpu"` and `dtype=torch.float32` for stability on the current hardware.

### Interpreting the Chronos Output

Chronos approaches time series differently from our classical and deep learning models:

- **It is univariate**: it sees only the historical log-price sequence, not macro variables or technical indicators
- **It is generative**: it produces a distribution of futures, not a single point prediction
- **It is zero-shot**: it never sees our specific dataset during training; it uses patterns learned from millions of other time series

This makes Chronos inherently a **structural baseline** rather than a tactical daily forecaster. It is well-suited to establishing a distributional view of plausible future paths over 5+ days, but it cannot capture conditional day-to-day signals from macro, FX, or technical features.

### Three-Model Volatility Comparison

The most analytically ambitious part of the notebook is a three-way volatility comparison using the last 5 days of the test set. The proxies used for each model are:

| Model | Volatility Proxy | Nature |
| --- | --- | --- |
| GARCH(1,1) | Forecast conditional volatility from Phase 4 | Proper conditional variance estimate |
| Hybrid LSTM | Absolute value of predicted residual: \|ε̂_t\| | Heuristic shock-magnitude proxy |
| Chronos | Prediction interval width: Q90 - Q10 | Structural uncertainty, not conditional variance |

This comparison should be read carefully. These three quantities are **not measuring the same thing**:

- GARCH produces a theoretically grounded conditional variance estimate updated at each timestep using the ARCH recursion
- The LSTM absolute-residual proxy is intuitive but is not a calibrated variance forecast
- The Chronos interval width reflects the model's epistemic uncertainty about a 5-day-ahead trajectory, not the conditional day-to-day realized volatility

The comparison is nonetheless instructive because it illustrates how different model families conceptualize uncertainty. The LSTM's multivariate context (including the GARCH estimate itself as a feature) likely helps it approximate short-term risk, while Chronos provides a structurally broader distributional view.

### Volatility Comparison Results

Over the 5-day comparison window:

1. **Hybrid LSTM achieves the lowest MAE against the realized volatility proxy**: having been trained on the residual series and given GARCH features as input, the LSTM has context about the current risk regime that Chronos does not.
2. **GARCH performs as a strong classical benchmark**: purpose-built for volatility clustering, it provides a robust and well-calibrated risk estimate despite using only the lagged residual series.
3. **Chronos shows the highest error against the realized proxy**: this is expected, not a failure. Chronos's interval width reflects structural distributional uncertainty, which is a wider and more conservative concept than the tight conditional variance estimated by GARCH.

### A Methodological Caution for Presentation

The 5-day evaluation window is statistically insufficient for stable metric ranking. With only 5 data points, the ranking of MAE values could reverse entirely with a different 5-day window. The correct claim is not "LSTM is definitively better than GARCH at volatility forecasting." The correct claim is:

"Over this specific 5-day period, the multivariate hybrid approach shows a lower error against the realized volatility proxy than the specialized conditional variance model. This is a directionally plausible result given the LSTM's richer feature set, but it cannot be generalized without a longer evaluation horizon."

For a rigorous comparison, one would need rolling volatility evaluation across at least 250+ days, analogous to what Phase 6 does for the mean models.

### What the Summary Table Means

| Feature | Hybrid LSTM | Amazon Chronos | GARCH(1,1) |
| --- | --- | --- | --- |
| Inputs | Multivariate: macro, technical, GARCH features | Univariate: historical price only | Univariate: past residuals and variances |
| Learning type | Supervised end-to-end training | Zero-shot generative | Statistical parametric |
| Core strength | Context-aware shock estimation | Structural uncertainty bounds | Volatility clustering capture |
| Best use case | Short-term tactical residual forecasting | Long-term distributional planning | Day-ahead conditional risk management |
| Output type | Point estimate of next residual | Distribution over future paths | Conditional variance σ²_t |

### Key Conclusions from the LSTM-Chronos Benchmark

1. **The hybrid STL + LSTM design is architecturally sound**: separating deterministic structure (STL trend and seasonal) from stochastic shocks (LSTM) is a principled strategy that lets the deep model concentrate on the hardest part of the prediction problem.
2. **Chronos provides a valuable structural reference**: its purpose is not to beat specialized models on daily conditional volatility. It provides an unbiased distribution of plausible trajectories, useful for scenario analysis and long-horizon uncertainty quantification.
3. **GARCH remains the appropriate daily risk benchmark**: its theoretical grounding and proper statistical calibration make it the right tool for conditional variance estimation, even when empirical proxies from a deep model may appear to perform better over short windows.
4. **The evaluation window matters**: a 5-day comparison is directionally informative but statistically fragile. Conclusions from this section should be presented with explicit acknowledgment of the small sample.

## VisualAnalytics Notebook Architecture

The `Assignment1/VisualAnalytics.ipynb` notebook is the visual companion to the entire six-phase pipeline. It reads only from `Assignment1/outputs/` and produces interactive Plotly charts organized in eight thematic blocks. It is the primary tool for presenting all model results to a non-specialist audience.

### Block 1: Data Health and Transformation Overview

This block answers: "What happened to the raw data before any model was fit?"

- **Row removal summary**: bar chart showing rows dropped before 2010-04-01 (55 rows) and non-trading rows removed (183 rows)
- **Normalized price trajectories**: all seven close prices are rebased to 100 at the first cleaned trading day (2010-04-01) using a **base-100 index**: $\text{Index}_t = (P_t / P_{t_0}) \times 100$, where $P_{t_0}$ is each asset's own price on the base date and `.iloc[0]` selects it. This is **not** a moving average or rolling-window calculation — it is a single division by a fixed starting value that removes the price-level difference between assets (NASDAQ at ~2,530, silver at ~17.5, etc.) so cumulative percentage growth is directly comparable on one chart. An index value of 200 means the asset doubled from its base-date price.
- The block makes the data cleaning decisions visible and auditable before any model result is shown

### Block 2: Log Returns, Correlation, and Stationarity

This block answers: "Why did we transform prices to returns, and does the transformation work?"

- **Return distribution histograms**: interactive histogram of log returns per asset, colored by asset, showing the fat-tailed nearly symmetric distributions
- **Return correlation heatmap**: lower-triangular heatmap of log-return correlations plus macro and FX variables — these are the cross-series dependencies used in the SARIMAX and PatchTST feature blocks
- **ADF evidence chart** (`-log10(p-value)` on y-axis, NOT raw p-values): the y-axis plots the negative base-10 logarithm of the ADF p-value, not the raw p-value itself. The threshold line sits at `-log10(0.05) ≈ 1.301`. A bar **above** the line means p < 0.05 → reject H₀ → stationary. A bar **below** the line means p > 0.05 → fail to reject H₀ → non-stationary (unit root present). Because of this `-log10` transformation, a taller bar correctly and unambiguously means stronger evidence against the unit-root null: price-level bars sit near zero (p ≈ 0.31–0.997 → `-log10` ≈ 0.001–0.48, far below the threshold) confirming non-stationarity; log-return bars tower at 20–27 (p ≈ 10⁻²⁰ to 10⁻²⁷, far above the threshold) confirming stationarity. This is entirely consistent with standard ADF theory as stated in the course material (Unit 4, p.37): H₀ = unit root present → non-stationary; H₁ = no unit root → stationary; reject H₀ when p < 0.05. The only thing that can seem counterintuitive is if one assumes the chart shows raw p-values — for raw p-values, a tall bar (high p) would indeed mean we **cannot** reject the unit root. The `-log10` transformation resolves this: it maps small p-values (strong evidence of stationarity) to large positive numbers, and large p-values (weak evidence of stationarity) to numbers near zero.

The key visual lesson from Block 2: price-level ADF bars stay below the threshold, while log-return bars are far above it. This is exactly the pattern that justifies the return transformation.

### Block 3: Technical-Indicator Dashboards

This block answers: "How do momentum, trend, and local volatility signals look per asset?"

For each of the seven assets, an interactive dashboard shows:

- **Bollinger Bands**: close price surrounded by the upper and lower bands, with shaded band width
- **RSI(14)**: momentum oscillator with horizontal reference lines at 70 (overbought) and 30 (oversold)
- **MACD histogram**: difference between the MACD line and its signal line, showing momentum acceleration or deceleration

These indicators are not used as standalone trading rules in this project. They are included as features in the PatchTST and LSTM models, where their numerical values become part of the multivariate input rather than binary signals.

### Block 4: STL Decomposition Dashboards

This block answers: "What is the structural composition of each asset's log-price series?"

- **Strength summary bars**: trend strength and seasonal strength side by side per asset. NASDAQ shows trend strength ≈ 1.00 and seasonal strength ≈ 0. Oil is the outlier at seasonal strength ≈ 0.39.
- **Seasonal amplitude bar chart**: shows that oil has a materially larger seasonal amplitude than all other assets, driven by a few extreme episodes rather than a stable weekly calendar pattern
- **Per-asset STL dashboard**: four subplots showing observed log price, trend, seasonal, and residual for each asset

The key lesson from Block 4: "These markets are predominantly trend-driven with very weak stable weekly seasonality. Modeling should focus on trend, autocorrelation, exogenous effects, and volatility clustering rather than deterministic seasonal terms."

### Block 5: Classical SARIMAX Baseline

This block answers: "What does the classical linear benchmark predict, and where does it fail?"

- **KPI summary indicators**: RMSE (0.01103), MAE (0.00824), and directional accuracy (52.78%) for the test period
- **Forecast timeline**: a continuous chart showing the training fitted values overlaid with the test forecast and a confidence band. The key visual: the forecast path is much smoother than the realized returns, and the confidence band widens correctly during high-uncertainty periods. The smoothness is not a flaw — it reflects that the model predicts the conditional mean, which is far less volatile than realized returns.
- **Coefficient chart with significance highlighting**: horizontal bar chart of the 9 SARIMAX parameters (8 exogenous + intercept). Reading the chart correctly requires understanding four elements together:
  - **X-axis**: estimated coefficient value (the magnitude of each predictor's effect on the conditional NASDAQ return, on the standardized scale)
  - **Y-axis**: the parameter names sorted by coefficient value
  - **Color coding**: blue bars are statistically significant at p < 0.05; red bars are not significant (confidence interval crosses zero)
  - **Thin black horizontal lines**: the 95% confidence interval for each coefficient. If the CI straddles the vertical dotted line at x = 0, the coefficient is statistically indistinguishable from zero — the corresponding predictor adds no reliable directional signal in this linear specification

  The empirical results from `outputs/phase3/coefficient_summary.csv`:

  | Parameter | Coefficient | p-value | Status |
  | --- | --- | --- | --- |
  | `intercept` | +0.000583 | 0.0132 | significant |
  | `sp500_ret_l1` | −0.001525 | 1.9×10⁻³⁰ | significant |
  | `gold_ret_l1` | +0.000354 | 0.0562 | not significant (borderline) |
  | `oil_ret_l1` | −0.0000280 | 0.924 | not significant |
  | `eur_usd_ret_l1` | +0.000345 | 0.175 | not significant |
  | `usd_chf_ret_l1` | +0.000328 | 0.183 | not significant |
  | `cpi_inflation_l1` | +0.000221 | 0.393 | not significant |
  | `rate_change_l1` | −0.000280 | 0.162 | not significant |
  | `gdp_growth_l1` | +0.0000673 | 0.837 | not significant |

  **Interpreting `sp500_ret_l1` (blue, negative, z = −11.47)**: This is the dominant predictor by a large margin. The negative sign reflects a short-horizon conditional mean-reversion tendency in the fitted regression: if yesterday's S&P 500 return is one training-standard-deviation above its mean, the model forecasts a slightly lower NASDAQ return today. This does **not** contradict the well-known strong positive contemporaneous correlation between SP500 and NASDAQ — that contemporaneous relationship is between same-day moves. The coefficient here is a lagged (day t−1 → day t) regression coefficient, which can have a different sign due to partial mean-reversion dynamics. The effect is statistically overwhelming (p ≈ 10⁻³⁰) but economically tiny: at ±0.0015, it shifts the forecast by roughly 0.15 log-return percentage points per standard-deviation move, which is small relative to typical daily volatility of ≈ 1%.

  **Interpreting the intercept (blue, positive, z = 2.48)**: Small positive drift term (≈ 0.058% per day), reflecting the secular upward trend in NASDAQ log returns over the 2010–2024 sample. Statistically significant because the sample is large enough to resolve a small constant mean.

  **Why 7 of 9 predictors are not significant**: This is the expected result under the Efficient Market Hypothesis. Daily financial returns have very low signal-to-noise ratio. Macro variables (GDP, CPI, interest rates) and FX rates operate on monthly or longer frequencies — their first-difference or log-growth at daily resolution is dominated by noise. The result is not a model failure; it is an honest empirical finding that the predictable structure in the conditional mean is very narrow. The SARIMAX order (0,0,0) selected by AIC reinforces this: the model correctly concludes that no additional autoregressive or moving-average terms improve fit after the exogenous block is added.

  **On coefficient magnitudes**: all exogenous variables are standardized before fitting. Coefficients are therefore comparable to each other in magnitude across predictors, but they are not in the original units of the variables (e.g., the `rate_change_l1` coefficient is not "basis points per basis point of rate move"; it is the NASDAQ log-return shift per one training-standard-deviation change in the lagged rate).

- **Residual diagnostic panel**: a 2×2 grid showing the residual time series (looking for structure — clusters of large errors confirm volatility clustering), histogram (heavy tails relative to Gaussian), Q-Q plot against normal (tail departures at both ends), and Ljung-Box p-values at lags 5, 10, 20. The Ljung-Box results are the critical output: lag-5 is borderline (p ≈ 0.059), but lags 10 and 20 strongly reject the no-autocorrelation null (p ≈ 10⁻¹¹). This means the residuals are not white noise — they still contain temporal structure, specifically volatility clustering. This is the direct empirical motivation for GARCH in Phase 4: the mean model is done, but the variance process still has structure that can be modeled.

The key lesson from Block 5: "The classical model captures the limited predictable structure in the conditional mean (mainly the SP500 equity linkage) but leaves residual autocorrelation and heavy-tail behavior behind. These are exactly the stylized facts that motivate Phase 4 (GARCH for volatility) and Phase 5 (deep learning for richer nonlinear mean structure)."

### Block 6: GARCH Volatility Diagnostics

This block answers: "Does the GARCH model successfully capture the volatility clustering in the residuals?"

- **KPI summary indicators**: persistence (0.9916), half-life (82.4 periods), volatility RMSE, and QLIKE. Persistence = α + β = 0.9916 means a volatility shock decays at rate 0.9916 per period, which is extremely slow (half-life = 82.4 trading days ≈ 4 months). This is a classic stylized fact for equity markets: volatility clustering is long-lived.

- **In-sample conditional volatility (σ̂_t time series)**: a smooth curve showing how the estimated daily standard deviation evolved over the full training period. Reading this chart: spikes upward mark stress episodes (March 2020 COVID crash, 2022 Fed rate hike cycle are clearly visible). Calm periods produce a low flat baseline. This is not the same as realized volatility (|ε_t|) — it is the model's latent estimate of the underlying variance process. The smoothness compared to |ε_t| is intentional: GARCH estimates the variance process, which is smoother than the individual shock realizations.

- **Out-of-sample volatility forecast vs realized proxy**: for the test period, the GARCH σ̂_t forecast (forward-projected from the last training-period state) is overlaid against the absolute SARIMAX residual |ε_t| as a realized volatility proxy. The key lesson: the GARCH curve tracks the *scale* of the actual residuals (it rises when residuals are large and falls when they are small) but cannot predict the sign of individual shocks. This is exactly what GARCH is designed to do — it models volatility, not direction.

- **Parameter bar chart (ω, α₁, β₁, ν with CIs)**: four bars with confidence intervals showing the fitted GARCH(1,1) Student-t parameters. Reading each:
  - **ω = 0.02733**: the long-run baseline variance contribution per period (on the scaled ×100 returns). Small relative to the overall variance, as expected when persistence is high.
  - **α₁ = 0.12767**: the ARCH term — how much last period's squared shock feeds into today's variance estimate. A value of ≈0.13 means recent large shocks have a moderate immediate impact on volatility.
  - **β₁ = 0.86396**: the GARCH term — how much yesterday's variance estimate carries over to today. At 0.86, most of the variance is inherited from the past state, creating the long memory.
  - **ν = 5.8907**: degrees of freedom for the Student-t innovation distribution. Values 4–8 indicate significantly fat tails. At ν ≈ 5.9, the distribution has finite variance but its tails are much heavier than Gaussian. Bars clearly away from zero with narrow CIs confirm all parameters are well-identified.

- **Ljung-Box diagnostic panel (the critical two-panel chart)**: two side-by-side plots of Ljung-Box p-values at multiple lags.
  - **Left panel — Ljung-Box on standardized residuals ε_t / σ̂_t**: p-values may still fall below the 5% line at some lags, indicating the standardized residuals still contain some serial dependence. This is the linear mean structure that GARCH does not model — it tells you the SARIMAX mean equation left some structure behind, which is consistent with the Phase 3 Ljung-Box finding.
  - **Right panel — Ljung-Box on squared standardized residuals (ε_t / σ̂_t)²**: p-values should all be clearly above the 5% threshold, meaning no significant autocorrelation in squared standardized residuals. This is the direct test of whether GARCH has absorbed the volatility clustering. When this panel shows p-values above 0.05 at all lags, GARCH has done exactly its job: the variance dynamics have been captured, even if some mean dynamics remain.

The key lesson from Block 6: "GARCH has successfully absorbed the volatility clustering: squared standardized residuals show no significant autocorrelation. The remaining serial dependence in the linear residuals is a motivation for richer nonlinear mean modeling, not a GARCH failure."

### Block 7: Deep Learning Benchmark (PatchTST-Style)

This block answers: "Does the deep model improve on the classical benchmark, and by how much?"

- **Glossary cells**: markdown cells defining lookback window, patch, stride, token, channel, RMSE, MAE, hit rate, and forecast-actual correlation — intended for oral explanation during class
- **Architecture description cells**: walkthrough of what `PatchTSTForecaster` and `PatchTSTDeepForecaster` do step by step
- **Provenance cell**: explicit statement that the architecture derives from the PatchTST paper (Nie et al., 2022) and official repository, not invented from scratch

- **Training curve (loss vs epoch)**: x-axis = training epoch (1–20+), y-axis = loss value, two lines: training loss and validation loss. Reading this correctly:
  - Both lines should decrease early — that's the model learning
  - Training loss continues to decrease monotonically
  - Validation loss flattens or begins to rise after the optimal epoch (epoch 9 here)
  - The gap between training and validation loss (if training < validation) is the overfitting gap
  - Best epoch = 9 means early stopping saved the model weights from epoch 9. All subsequent epochs would have reduced training loss but increased generalization error. The chart makes this visually unambiguous.

- **Forecast timeline**: the full test period on the x-axis, NASDAQ log-return on the y-axis, with two lines: realized returns (jagged, high-amplitude) and PatchTST forecast (smooth, compressed amplitude). The key visual: the forecast never produces extreme spike predictions — it stays close to zero. This is not a model failure. It is a fundamental consequence of MSE-optimal forecasting when signal-to-noise is low: the MSE-optimal point forecast of a near-zero conditional mean is a near-zero value. The model correctly hedges toward the center rather than guessing on extreme moves.

- **3-panel comparison** (the most diagnostic set of charts):
  - **Panel 1 — Actual vs Forecast scatter**: each point is one test-period day, x = forecast, y = realized return. A perfect model would show tight points along the diagonal (y=x line). In practice, the cloud is wide and weakly elongated along the diagonal. The positive slope of the best-fit line confirms positive predictive alignment. The wide cloud confirms the model explains only a small fraction of return variance. A cloud of this shape is the expected outcome for a well-calibrated financial return model — do not expect a tight line.
  - **Panel 2 — Return distribution comparison**: histogram or density of PatchTST forecasts vs histogram of realized returns. The realized distribution is wider and heavier-tailed. The forecast distribution is narrow and approximately centered at zero. This compression is again MSE-optimal behavior: when the signal is weak, the optimal strategy is to forecast close to the mean. If the forecast and realized distributions looked identical, that would be suspicious (the model would be predicting extreme shocks that are unpredictable by construction).
  - **Panel 3 — Rolling directional accuracy**: sliding window directional accuracy through the test period. This shows how the model's hit rate evolves over time. Regime dependence is visible: there are periods where accuracy exceeds 60% and periods where it falls near 50% (random). Neither model dominates consistently across all time segments.

- **Metric comparison table**: RMSE, MAE, and hit rate for PatchTST (0.01090 / 0.00810 / 57.1%) vs Phase 3 SARIMAX (0.01103 / 0.00824 / 52.8%). PatchTST wins on all three static metrics. The magnitude of improvement is modest — 1.2% RMSE reduction, 4.3 percentage-point gain in hit rate. The table poses the key question for Block 8: does this statistical improvement translate into better trading economics?

The key lesson from Block 7: "The deep model improves all metrics but only modestly. Forecast distribution remains conservative and centered near zero. This is realistic for daily financial return prediction with low signal-to-noise ratio. The 3-panel comparison reveals that the model's apparent conservatism is optimal behavior under MSE loss, not a tuning failure."

### Block 8: Rolling Backtesting and Market Frictions

This block answers: "Does the statistical improvement in Phase 5 survive repeated historical evaluation under trading costs?"

- **KPI summary panel**: side-by-side snapshot of Sharpe ratio, maximum drawdown, net cumulative return, and average turnover for SARIMAX vs PatchTST. The summary already reveals the main finding before opening any chart:

  | Metric | SARIMAX | PatchTST |
  | --- | --- | --- |
  | Sharpe ratio | **0.525** | 0.496 |
  | Net cumulative return | **+65.8%** | +59.5% |
  | Maximum drawdown | −40.4% | −42.6% |
  | Average turnover | 0.687 | **0.226** |

  SARIMAX wins on Sharpe and net return despite worse statistical metrics. PatchTST wins on turnover (it trades less frequently).

- **Cumulative wealth paths**: the x-axis spans the full backtest period (rolling folds), the y-axis shows portfolio value starting at 1.0. Two lines: SARIMAX wealth curve and PatchTST wealth curve. Reading this chart:
  - A line that stays above 1.0 throughout means the strategy made money overall
  - Drawdown periods are visible as V-shapes or extended flats — both models have a material drawdown episode (maximum ≈ −40% for SARIMAX, −43% for PatchTST), which is a critical risk flag
  - The slope of the line in later folds shows whether model skill is holding up — if the wealth curve flattens in fold 4 or 5, the model has lost its edge in that regime
  - SARIMAX ending higher than PatchTST is the core empirical result: better statistical fit did not produce better economic performance

- **Fold-by-fold directional accuracy**: bar chart with 5 groups (one per fold), each group containing two bars (SARIMAX and PatchTST). Reading this:
  - No model achieves consistently high accuracy across all five folds — accuracy swings between ~45% and ~65% depending on the market regime in each fold
  - In some folds, SARIMAX has higher directional accuracy; in others, PatchTST does — neither model dominates consistently
  - This chart is the key evidence that model skill is regime-dependent: the fold-by-fold variability is as large as the overall difference between the two models

- **Turnover comparison**: bar chart showing average daily turnover per fold and in aggregate. Turnover measures the fraction of the portfolio that is rebalanced each day — higher turnover means more trades, more transaction costs (5 bps round-trip per trade), more cost drag. PatchTST's lower turnover (0.226 vs 0.687) means SARIMAX is changing its position more frequently. Despite this disadvantage, SARIMAX still outperforms on net return — which means the SARIMAX signals are worth paying the additional transaction costs for, while PatchTST's more conservative signals are not valuable enough to compensate for even the smaller cost it does incur.

The key lesson from Block 8: "Statistical improvement in forecast metrics does not automatically translate into better trading performance. SARIMAX's higher turnover and lower hit rate are offset by the economic value of the signals it generates. Once transaction costs and regime variability are accounted for, the classical benchmark is the better trading strategy, not the deep learning one. The lesson: evaluate models economically, not just statistically."

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

$$
P_t \approx T_t \times S_t \times E_t
$$

where $T_t$ is trend, $S_t$ is a seasonal factor, and $E_t$ is an irregular component. Taking logs converts this into an additive representation:

$$
\log P_t \approx \log T_t + \log S_t + \log E_t.
$$

STL is an additive decomposition method, so applying it to log prices is much more coherent than applying it directly to level prices whose variability scales with their magnitude.

### STL mechanics

STL means Seasonal-Trend decomposition using Loess. It models a series as

$$
y_t = T_t + S_t + R_t
$$

where:

- $T_t$ is the smooth trend,
- $S_t$ is the repeating seasonal component,
- $R_t$ is the remainder or residual.

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

$$
F_T = \max\left(0, 1 - \frac{\operatorname{Var}(R_t)}{\operatorname{Var}(T_t + R_t)}\right)
$$

for trend strength and

$$
F_S = \max\left(0, 1 - \frac{\operatorname{Var}(R_t)}{\operatorname{Var}(S_t + R_t)}\right)
$$

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

### What each STL component means in plain language

For a given asset, the STL decomposition produces three interpretable pieces:

- **Trend**: the slow-moving backbone of the series
- **Seasonality**: the repeating component at the chosen period
- **Residual**: what is left after removing trend and seasonality

An intuitive way to explain them is:

- the **trend** is the direction the series would follow if short-term fluctuations were smoothed away,
- the **seasonal** component is the part that repeats on a regular calendar pattern,
- the **residual** is the surprise component: shocks, dislocations, and local irregularity not explained by the first two parts.

If the seasonal component stays close to zero and does not show a stable repeating shape, that is evidence that the chosen seasonal cycle is weak.

### Example: how to explain the NASDAQ STL plot

NASDAQ is the cleanest example in this project.

Its summary metrics are approximately:

- trend strength: `0.9996`
- seasonal strength: `0.0000`
- residual share of variance: `0.0004`

A rigorous interpretation is:

"For NASDAQ log prices, the smooth trend explains almost all of the non-noise structure detected by STL. The weekly seasonal component is essentially absent under the chosen 5-trading-day cycle, and the residual component is small relative to the total variance."

If a teacher asks what that looks like visually, the answer is:

- the observed log-price series and the trend line sit very close to each other,
- the seasonal line oscillates tightly around zero without a stable repeated weekday pattern,
- and the residual line mainly captures shock days and short-lived deviations from the smooth path.

### Why zero seasonal strength does not mean "no short-term movement"

This point is easy to misunderstand.

When seasonal strength is near zero, it does **not** mean:

- the market is flat,
- the market has no volatility,
- or the market has no short-run fluctuations.

It means only that the fluctuations are **not well described by a stable repeating pattern at the chosen seasonal period**.

So a market can still be very volatile and yet have essentially no stable weekly seasonality. That is exactly what happens for several financial assets in this project.

### Asset-by-asset interpretation of the STL results

#### NASDAQ

- Trend strength: `0.9996`
- Seasonal strength: `0.0000`
- Residual share of variance: `0.0004`

Interpretation:

- the dominant structure is a persistent long-run trend,
- there is no evidence of a stable weekly seasonal pattern,
- and the residual component is comparatively small.

#### S&P 500

- Trend strength: `0.9993`
- Seasonal strength: `0.0000`
- Residual share of variance: `0.0007`

Interpretation:

- very similar to NASDAQ,
- extremely trend-dominated in log-price space,
- with weak residual noise relative to the trend,
- and no material weekly seasonality.

#### Gold

- Trend strength: `0.9973`
- Seasonal strength: `0.0000`
- Residual share of variance: `0.0027`

Interpretation:

- gold still looks strongly trend-driven,
- the residual variation is larger than for NASDAQ and the S&P 500 but still modest,
- and the STL seasonal component does not support a stable weekly cycle.

#### Silver

- Trend strength: `0.9947`
- Seasonal strength: `0.0000`
- Residual share of variance: `0.0053`

Interpretation:

- silver remains trend-dominated,
- but among the assets studied it has one of the larger residual shares,
- meaning short-term irregular movements are relatively more important than in the equity indices.

#### Platinum

- Trend strength: `0.9965`
- Seasonal strength: `0.0000`
- Residual share of variance: `0.0035`

Interpretation:

- trend is still the dominant component,
- residual shocks matter more than in NASDAQ or the S&P 500,
- but the decomposition still does not support strong weekly seasonality.

#### Palladium

- Trend strength: `0.9976`
- Seasonal strength: `0.0000`
- Residual share of variance: `0.0024`

Interpretation:

- palladium exhibits a strong smooth trend,
- weak stable seasonality,
- and a moderate shock component that is visible in the residual series.

#### Oil

- Trend strength: `0.9986`
- Seasonal strength: `0.3941`
- Seasonal amplitude: `1.7136`
- Residual share of variance: `0.0014`

Oil is the only asset that looks different numerically, but this must be interpreted carefully.

The seasonal-strength score is materially positive, yet the weekday-average seasonal values remain close to zero. This means the larger seasonal metric is not behaving like a clean, stable "Monday effect" or "Friday effect." Instead, the seasonal component is being influenced by a few extreme episodes, which is why its amplitude is much larger than the other assets.

The rigorous conclusion is:

- oil clearly contains strong trend structure,
- the weekly seasonal metric is unstable and should not be over-interpreted as a true persistent calendar effect,
- and visual inspection is necessary to avoid overstating what the summary statistic alone seems to suggest.

### How to explain the STL visuals in class

If you need to explain the decomposition plots orally, a good structure is:

1. Start with the observed log-price panel and say whether the asset looks trend-dominated.
2. Look at the seasonal panel and ask whether it shows a clear repeating shape around the 5-day trading cycle.
3. Look at the residual panel and explain whether the remaining movement is mostly small noise or whether there are noticeable shock episodes.

For most assets in this project, the correct explanation is:

"The decomposition shows a strong long-run trend, almost no stable weekly seasonal pattern, and a residual component that captures shocks and irregular moves rather than systematic calendar repetition."

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

$$
\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^{p}\phi_i \Delta y_{t-i} + \varepsilon_t.
$$

The null hypothesis is that the series has a unit root, which implies non-stationarity. Rejecting the null supports stationarity. In finance, raw prices are usually non-stationary, while returns are often closer to stationary.

### STL Decomposition

STL decomposes a series into:

$$
y_t = T_t + S_t + R_t
$$

where $T_t$ is trend, $S_t$ is seasonal structure, and $R_t$ is residual noise. This helps isolate smoother long-run movement from recurring seasonal patterns and idiosyncratic shocks.

In this project, STL is applied to log prices on the trading-day calendar with a period of 5 observations. Empirically, the decomposition shows very strong trend structure and very weak seasonal structure for most assets.

### ARMA / ARIMA / SARIMAX

An ARMA model combines autoregressive and moving-average dynamics:

$$
y_t = \sum_{i=1}^{p}\phi_i y_{t-i} + \varepsilon_t + \sum_{j=1}^{q}\theta_j \varepsilon_{t-j}.
$$

ARIMA extends this by differencing to handle non-stationarity. SARIMAX further allows seasonal structure and exogenous regressors, which is appropriate for financial settings where macro variables may help explain returns or transformed prices.

In the implemented benchmark, AIC-driven `auto_arima` selected a non-seasonal `(0,0,0)` mean specification once lagged exogenous variables were included, which suggests very limited additional ARMA structure in the target return series.

### GARCH(1,1)

After modeling the conditional mean, the conditional variance can be modeled as

$$
\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2.
$$

This captures volatility clustering, a core stylized fact of financial returns. In this project, GARCH is fit on the ARIMA residuals to separate mean forecasting from risk forecasting.

### Deep Learning / Foundation Forecasting

The implemented deep benchmark in this project is a PatchTST-style transformer. Chronos-2 remains relevant as a foundation-model reference point, but it is treated as an extension rather than the first local implementation.

#### PatchTST-style forecasting

PatchTST is built around the idea that long univariate or multivariate histories can be broken into local temporal patches before entering a transformer. If the lookback window is $L$, patch length is $P$, and stride is $S$, then the number of patches is

$$
N_{patch} = 1 + \frac{L - P}{S}.
$$

Each patch is embedded into a latent vector and processed by self-attention. This lets the model compare local historical fragments rather than isolated points. In finance, those fragments may correspond to short bursts of momentum, reversals, consolidations, or shock-recovery patterns.

The implemented Phase 5 model uses:

- a 60-day lookback,
- 10-day patches,
- 5-day stride,
- and a transformer encoder to forecast the next NASDAQ log return from lagged multivariate features.

The motivation is that nonlinear interactions between assets, technical indicators, FX variables, and macro changes may matter in a way that a linear SARIMAX cannot capture.

#### Why Chronos-2 is discussed but not used as the first implemented model

Chronos-2 belongs to the foundation-model family of time-series forecasting. Conceptually, such models are pretrained on broad corpora and then used in zero-shot or lightly adapted forecasting tasks. That makes them attractive as modern benchmarks.

However, for this assignment, PatchTST is the more defensible first implementation because:

- it is fully trainable and reproducible from local project data,
- its feature and training pipeline are transparent,
- and its comparison against the classical baseline is easier to interpret scientifically.

So the deep-learning phase is not meant to "replace" statistical modeling blindly. It is meant to test whether a modern nonlinear sequence learner can produce incremental predictive gains under a fair and reproducible setup.

### Backtesting and Financial KPIs

Forecasting performance will be evaluated not only with error metrics, but also with finance-specific metrics:

- Hit Rate: directional accuracy of predicted returns,
- Sharpe Ratio: return per unit of volatility,
- Maximum Drawdown: worst cumulative loss from a peak,
- and transaction-cost-aware PnL under slippage and commissions.

### Technical Indicators Used in Phase 1

#### Relative Strength Index (RSI)

RSI is a bounded momentum oscillator. If $\Delta P_t = P_t - P_{t-1}$, then gains and losses are separated as

$$
G_t = \max(\Delta P_t, 0), \qquad L_t = \max(-\Delta P_t, 0).
$$

Using Wilder-style smoothed averages over a window $n$,

$$
RS_t = \frac{\text{AvgGain}_t}{\text{AvgLoss}_t}
$$

and

$$
RSI_t = 100 - \frac{100}{1 + RS_t}.
$$

RSI helps detect overbought and oversold conditions, but in this project it is treated as a quantitative feature rather than a standalone trading rule.

#### MACD

MACD measures trend and momentum by comparing fast and slow exponential moving averages:

$$
MACD_t = EMA_{12}(P_t) - EMA_{26}(P_t).
$$

The signal line is a smoothed version of MACD:

$$
Signal_t = EMA_{9}(MACD_t),
$$

and the histogram is

$$
Hist_t = MACD_t - Signal_t.
$$

These quantities help quantify whether price momentum is accelerating or decelerating.

#### Bollinger Bands

For a rolling window $n$, Bollinger Bands are defined as

$$
Middle_t = \frac{1}{n}\sum_{i=0}^{n-1} P_{t-i},
$$

$$
Upper_t = Middle_t + k \sigma_t,
$$

$$
Lower_t = Middle_t - k \sigma_t,
$$

where $\sigma_t$ is the rolling standard deviation and $k$ is usually set to 2. The bands provide a normalized way to measure how far a price has moved relative to its recent local volatility.

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

## Model Summary Table

| Phase | Model / Procedure | Target | Exact Specification | Purpose | Key Reason |
| --- | --- | --- | --- | --- | --- |
| Phase 3 | SARIMAX classical baseline | `nasdaq log_return` | Order `(0,0,0)`, seasonal `(0,0,0,0)`, constant, 8 lagged exogenous regressors | Conditional mean benchmark | Interpretable, statistically grounded, necessary before claiming value from complex models |
| Phase 4 | GARCH(1,1) volatility model | Phase 3 residuals | GARCH(1,1) with Student-t innovations, scaled by 100 for numerical stability | Conditional variance / risk benchmark | Returns exhibit volatility clustering and heavy tails; GARCH is the standard classical risk model |
| Phase 5 | PatchTST-style transformer | `nasdaq log_return` | 60-day lookback, 10-day patches, 5-day stride, 33 lagged multivariate features | Nonlinear sequence forecasting benchmark | Inspired by PatchTST paper; tests whether transformer attention improves over the linear baseline |
| Phase 6 | Rolling walk-forward backtest | `nasdaq log_return` | 5-fold backtest, sign-based trading, 2 bps commissions, 3 bps slippage | Economic evaluation under frictions | Forecast accuracy alone is not sufficient; models must survive repeated historical evaluation under costs |
| LSTM-Chronos | Hybrid STL-LSTM + zero-shot Chronos | `sp500 STL residual` | 2-layer LSTM (hidden=64, seq=30); Chronos-T5-base zero-shot (5-step, 80% CI) | Alternative deep benchmark with foundation model validation | Tests whether targeting the stochastic residual directly (rather than the raw return) improves predictability; Chronos provides unbiased structural reference |

### Performance Summary

| Model | Target | RMSE | MAE | Hit Rate | Notes |
| --- | --- | --- | --- | --- | --- |
| SARIMAX (Phase 3, single split) | nasdaq log_return | 0.01103 | 0.00824 | 52.8% | Linear regression on lagged exogenous block |
| PatchTST (Phase 5, single split) | nasdaq log_return | 0.01090 | 0.00810 | 57.1% | Modest but real improvement over SARIMAX |
| SARIMAX (Phase 6, rolling 5-fold) | nasdaq log_return | 0.01612 | 0.01152 | 51.8% | Sharpe 0.525, net return +65.8%, MaxDD -40.4% |
| PatchTST (Phase 6, rolling 5-fold) | nasdaq log_return | 0.01597 | 0.01123 | 53.5% | Sharpe 0.496, net return +59.5%, MaxDD -42.6% |
| Hybrid LSTM (LSTM-Chronos notebook) | sp500 STL residual | — | Lowest over 5-day window | — | Different target; not directly comparable to above |

## Current Project State

All six phases of the structured pipeline are implemented and validated:

- **Phase 1**: Preprocessing and stationarity code validated on the live dataset. ADF tests confirm price-level non-stationarity and log-return stationarity for all seven assets. Outputs in `Assignment1/outputs/phase1/`.
- **Phase 2**: STL decomposition implemented, exported, and interpreted. Strong trend structure, weak weekly seasonality confirmed for six of seven assets. Outputs in `Assignment1/outputs/phase2/`.
- **Phase 3**: SARIMAX classical benchmarking with `pmdarima` AIC-driven order selection, full residual diagnostics, and coefficient analysis. Selected order `(0,0,0)`. Outputs in `Assignment1/outputs/phase3/`.
- **Phase 4**: GARCH(1,1) volatility modeling on Phase 3 residuals with Student-t innovations. Persistence 0.9916, half-life 82.4 periods. Squared standardized residual autocorrelation resolved. Outputs in `Assignment1/outputs/phase4/`.
- **Phase 5**: PatchTST-style transformer implemented, trained with early stopping (best epoch 9), and compared against Phase 3 baseline. Modest improvement on all three metrics. Outputs in `Assignment1/outputs/phase5/`.
- **Phase 6**: 5-fold rolling walk-forward backtest with market frictions. Key finding: PatchTST is statistically slightly better but economically not dominant; SARIMAX remains competitive on Sharpe and net return. Outputs in `Assignment1/outputs/phase6/`.

Additional work implemented in standalone notebooks:

- **`01_EDA.ipynb`**: Deep exploratory analysis establishing mixed-frequency structure, data quality, distribution behavior, and cross-asset correlations.
- **`03_Forecasting_LSTM_&_Chronos.ipynb`**: Hybrid STL-LSTM benchmark targeting SP500 residuals, combined with Amazon Chronos-T5 zero-shot validation and three-model volatility comparison.
- **`VisualAnalytics.ipynb`**: Interactive Plotly companion covering all eight analytical blocks from data health through rolling backtesting.

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
12. Nie, Y., Nguyen, N. H., Sinthong, P., and Kalagnanam, J. (2022). *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*. arXiv:2211.14730.
13. Official PatchTST repository by Yuqi Nie et al.: `https://github.com/yuqinie98/PatchTST`.
14. Sharpe, W. F. (1994). The Sharpe Ratio.
15. Ansari, A. F., et al. (2024). *Chronos: Learning the Language of Time Series*. Amazon Science. arXiv:2403.07815.
16. Hochreiter, S., and Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.
