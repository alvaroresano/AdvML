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

### Raw prices vs Returns 

Raw prices are usually non-stationary: their mean and variance structure drift (change) over time and generally exhibit exponential growth trajectories over long horizons, which breaks the assumptions behind ARIMA-class models and weakens statistical inference. Therefore, that's why we choose returns.

Simple returns measure the straightforward percentage change between two periods and are calculated as the difference between the current and previous price divided by the previous price. While intuitive and asset-additive—meaning the return of a portfolio is the weighted sum of the simple returns of its constituent assets—they present significant statistical drawbacks for time series modeling. Simple returns are bounded below by -100%, as an asset's price cannot drop below zero, but they are theoretically unbounded on the upside, leading to an asymmetric distribution. Furthermore, the product of normally distributed variables does not yield a normal distribution, complicating probabilistic modeling.

### Why log returns instead of simple returns

Logarithmic returns, calculated as the natural logarithm of the ratio of the current price to the previous price, are the standard for advanced time series forecasting. They offer several mathematical advantages. First, log returns compound additively over time, meaning the cumulative log return over a sequence of periods is simply the sum of the individual period log returns. Second, they possess perfect symmetry. If an asset's price increases from a baseline to a higher value and then returns to the baseline, the positive and negative log returns are absolute equals, unlike simple returns which mathematically distort the recovery required from a drawdown. Finally, log returns map the domain of positive real numbers to the entire real line, aligning perfectly with the assumptions of standard regression models and Gaussian error distributions. If asset prices follow a geometric Brownian motion, their log returns are normally distributed

For a price series $$(P_t)$$, the simple return is:

$$
[
R_t = \frac{P_t - P_{t-1}}{P_{t-1}}
]$$

and the log return is

$$[
r_t = \log\left(\frac{P_t}{P_{t-1}}\right) = \log(1 + R_t).
]$$

Log returns are usually preferred in quantitative modeling because:

- they are additive over time, which simplifies aggregation and modeling,
- they often stabilize variance better than raw prices,
- and many financial models are formulated naturally in continuously compounded returns.

For small returns, log returns and simple returns are numerically close, since $$(log(1+R_t) \approx R_t)$$

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

### ARIMA intuition

An ARIMA model is built from three pieces:

- **AR(p)**: autoregressive terms, meaning the series depends on its own past values,
- **I(d)**: integration or differencing, used to remove non-stationarity,
- **MA(q)**: moving-average terms, meaning the series depends on past shocks.

In compact notation:

$$[
\phi(B)(1-B)^d y_t = c + \theta(B)\varepsilon_t
]$$

where $$(B)$$ is the lag operator, $$(\phi(B))$$ is the autoregressive polynomial, and $$(\theta(B))$$ is the moving-average polynomial.

For financial returns, differencing is often unnecessary because returns are already much closer to stationary than price levels. That expectation was already supported by the ADF results from Phase 1.

### What SARIMAX adds

SARIMAX extends ARIMA by allowing:

- optional seasonal structure,
- and exogenous regressors.

Its general idea is:

$$[
y_t = c + \beta^\top x_t + \text{ARMA dynamics} + \varepsilon_t.
]$$

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

$$[
AIC = 2k - 2\log L
]$$

where $$(k)$$º is the number of estimated parameters and $$(L)$$ is the maximized likelihood.

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

### The GARCH(1,1) model

Let $$(\varepsilon_t)$$ be the residual from the Phase 3 mean model. A GARCH(1,1) model writes the conditional variance as

$$[
\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2.
]$$

Each term has a clear interpretation:

- $$(\omega)$$: the long-run variance floor
- $$(\alpha)$$: how strongly new shocks move current variance
- $$(\beta)$$: how persistent variance is over time

If $$(\alpha)$$ is large, volatility reacts strongly to new information. If $$(\beta)$$ is large, volatility decays slowly after a shock. The quantity

$$[
\alpha + \beta
]$$

is called **persistence**. When this sum is close to 1, volatility is highly persistent.

### Why the model uses zero mean

The GARCH model is fit on the residuals from the SARIMAX mean model. Since those residuals are already what remains after modeling the conditional mean, it is coherent to use a **zero-mean** volatility model in Phase 4.

In plain language:

- Phase 3 already tried to explain the average direction,
- so Phase 4 focuses only on the changing scale of uncertainty around zero.

### Why a Student-t distribution is used

The Phase 3 Jarque-Bera and Q-Q diagnostics already showed heavy tails. A Gaussian innovation assumption would therefore be too restrictive.

For this reason, the Phase 4 implementation uses a Student-t innovation distribution. This allows the model to accommodate heavier tails than a normal distribution.

The Student-t distribution has a degrees-of-freedom parameter, here denoted by \(\nu\). Smaller \(\nu\) means heavier tails. As \(\nu \to \infty\), the Student-t approaches a Gaussian distribution.

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

- a realized squared error proxy, \(\varepsilon_t^2\),
- forecast variance RMSE,
- forecast volatility RMSE,
- and QLIKE.

The QLIKE loss is

\[
QLIKE_t = \log(\hat{\sigma}_t^2) + \frac{\varepsilon_t^2}{\hat{\sigma}_t^2}.
\]

This loss is widely used in volatility forecasting because it is more robust than plain squared loss when realized variance is noisy.

### Empirical findings from Phase 4

The fitted model is:

- GARCH(1,1)
- innovation distribution: Student-t
- target series: Phase 3 NASDAQ mean residuals

Estimated parameters:

- \(\omega \approx 0.02733\)
- \(\alpha_1 \approx 0.12767\)
- \(\beta_1 \approx 0.86396\)
- \(\nu \approx 5.8907\)

Derived quantities:

- persistence: \(\alpha_1 + \beta_1 \approx 0.9916\)
- unconditional volatility: approximately `0.0181`
- volatility half-life: approximately `82.42` trading periods

### How to explain persistence and half-life

Persistence near 1 means volatility shocks fade slowly.

The half-life is the number of periods needed for a volatility shock to decay by half. A value around `82` trading days means that elevated volatility can remain influential for several months.

That is an economically plausible result for financial markets, especially after major macro or risk-off episodes.

### What the parameter estimates mean in plain language

The fitted coefficients say:

- volatility responds meaningfully to new shocks because \(\alpha_1\) is clearly positive,
- volatility is highly persistent because \(\beta_1\) is very large,
- and the tail behavior is materially heavier than Gaussian because \(\nu\) is finite and relatively low.

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

\[
N_{patch} = 1 + \frac{L - P}{S}
\]

where:

- \(L = 60\) is the lookback window,
- \(P = 10\) is the patch length,
- \(S = 5\) is the stride.

So here:

\[
N_{patch} = 1 + \frac{60 - 10}{5} = 11.
\]

This means the transformer processes 11 learned temporal fragments rather than 60 single-day points.

The intuition is simple:

- a patch can represent a short market episode,
- such as a small drawdown, rebound, momentum burst, or sideways consolidation,
- and attention can then learn which of those historical fragments matter most for the next-day forecast.

### Mathematical structure of the implemented model

Let the multivariate input window be

\[
X_t \in \mathbb{R}^{L \times C},
\]

where:

- \(L = 60\) is the lookback length,
- \(C = 33\) is the number of input features.

Each channel is patched along the time axis into segments of length \(P = 10\). Each patch is projected through a learned linear map into a latent representation of dimension \(d_{model} = 32\).

If a patch vector is \(x_{patch} \in \mathbb{R}^{10}\), the embedded token is

\[
z_{patch} = W x_{patch} + b,
\]

where \(W \in \mathbb{R}^{32 \times 10}\).

Positional embeddings are added so the model can distinguish where each patch lies in the lookback window. The embedded patch sequence is then passed through a transformer encoder.

The transformer attention mechanism computes weights of the form

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
\]

Intuitively:

- \(Q\) asks what the current token is looking for,
- \(K\) describes what each historical token contains,
- \(V\) carries the information to be aggregated.

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

The log return between prices \(P_{t-1}\) and \(P_t\) is

\[
r_t = \log\left(\frac{P_t}{P_{t-1}}\right).
\]

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

\[
z_t = \frac{P_t - \mu_t}{\sigma_t},
\]

where:

- \(P_t\) is the current price,
- \(\mu_t\) is the rolling 20-day mean,
- \(\sigma_t\) is the rolling 20-day standard deviation.

This tells us how many rolling standard deviations the current price is above or below its local average.

Examples:

- \(z_t = 0\): price is exactly at the local rolling mean,
- \(z_t = 2\): price is about two local standard deviations above the mean,
- \(z_t = -1.5\): price is about one and a half local standard deviations below the mean.

This is often easier to explain to technical audiences than the raw Bollinger bands because it is a standardized distance measure.

### Example of the leakage logic

Suppose the model predicts the NASDAQ return for trading day \(t\).

It uses:

- the previous 60 days of lagged returns,
- the previous 60 days of lagged technical indicators,
- the previous 60 days of lagged macro-change and FX features.

It does **not** use day-\(t\) technical indicators or day-\(t\) close information when forecasting day \(t\)'s return.

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

\[
RMSE = \sqrt{\frac{1}{n}\sum_{t=1}^{n}(y_t - \hat{y}_t)^2 }.
\]

It penalizes large forecast errors more strongly than small ones because the errors are squared before averaging.

A lower RMSE is better.

#### MAE

The **Mean Absolute Error** is

\[
MAE = \frac{1}{n}\sum_{t=1}^{n}|y_t - \hat{y}_t|.
\]

It measures the average absolute forecast miss without squaring the errors.

A lower MAE is better.

#### Hit rate / directional accuracy

The **hit rate** checks whether the model got the sign of the return correct:

\[
HitRate = \frac{1}{n}\sum_{t=1}^{n}\mathbf{1}\{\text{sign}(y_t)=\text{sign}(\hat{y}_t)\}.
\]

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

#### `PatchTSTForecaster`

This is the neural network itself.

Its constructor creates:

- `patch_embedding`: converts each 10-day patch into a 32-dimensional latent vector,
- `position_embedding`: tells the model where each patch sits in the lookback window,
- `encoder`: the transformer stack,
- `channel_norm`: stabilizes the pooled representation,
- `head`: converts the encoded representation into one scalar forecast.

The `forward` pass does the following:

1. Transpose the input from `[batch, time, features]` to `[batch, features, time]`.
2. Create overlapping temporal patches with `unfold`.
3. Project each patch into the latent space with `patch_embedding`.
4. Add positional embeddings.
5. Reshape the patch sequence so the transformer encoder can process it.
6. Run self-attention and feedforward layers through the encoder.
7. Average across patches to get one summary per channel.
8. Flatten the channel summaries and map them through the prediction head.
9. Output one next-step forecast.

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

For each target day \(t\), the model input is the block of features from days \(t-60\) through \(t-1\), and the label is the target return at day \(t\).

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

\[
position_t = \text{sign}(\hat{r}_t)
\]

where:

- \(+1\) means take a long position,
- \(-1\) means take a short position,
- \(0\) means no directional conviction if the forecast is exactly zero.

The gross strategy return is

\[
R^{gross}_t = position_t \cdot r_t,
\]

where \(r_t\) is the realized NASDAQ log return.

### Market frictions: commissions and slippage

The backtest includes:

- commission = `2` basis points,
- slippage = `3` basis points.

So the total trading cost rate is `5` basis points per unit turnover.

Turnover is defined as

\[
turnover_t = |position_t - position_{t-1}|.
\]

This is important because flipping from long to short is more expensive than staying long.

The transaction cost is

\[
cost_t = turnover_t \cdot c,
\]

where \(c = 0.0005\).

The net strategy return is therefore

\[
R^{net}_t = R^{gross}_t - cost_t.
\]

### Why transaction costs matter

A model that trades too often can look good statistically but poor economically. If every small forecast change forces position changes, the apparent predictive edge can be consumed by costs.

That is why Phase 6 reports turnover explicitly.

### Financial KPI glossary

#### Net cumulative return

If the strategy net return on day \(t\) is \(R^{net}_t\), cumulative wealth is

\[
W_t = \prod_{i=1}^{t}(1 + R^{net}_i).
\]

The net cumulative return is \(W_T - 1\).

#### Sharpe ratio

The annualized Sharpe ratio used here is

\[
Sharpe = \sqrt{252}\frac{\bar{R}^{net}}{\sigma(R^{net})}.
\]

It measures return per unit of realized variability. A higher Sharpe ratio is better.

#### Maximum drawdown

Maximum drawdown measures the worst peak-to-trough loss in the cumulative wealth path:

\[
Drawdown_t = \frac{W_t}{\max_{s \le t} W_s} - 1.
\]

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

### Why this can happen

A model can improve forecasting metrics without improving trading outcomes for several reasons:

1. the forecast improvement may be too small relative to market noise,
2. the improved predictions may not occur on the most economically important days,
3. the forecast amplitudes may be too conservative to generate larger strategy gains,
4. regime sensitivity may reduce the consistency of the edge through time.

That is exactly why backtesting cannot be replaced by RMSE tables alone.

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

This captures volatility clustering, a core stylized fact of financial returns. In this project, GARCH is fit on the ARIMA residuals to separate mean forecasting from risk forecasting.

### Deep Learning / Foundation Forecasting

The implemented deep benchmark in this project is a PatchTST-style transformer. Chronos-2 remains relevant as a foundation-model reference point, but it is treated as an extension rather than the first local implementation.

#### PatchTST-style forecasting

PatchTST is built around the idea that long univariate or multivariate histories can be broken into local temporal patches before entering a transformer. If the lookback window is \(L\), patch length is \(P\), and stride is \(S\), then the number of patches is

\[
N_{patch} = 1 + \frac{L - P}{S}.
\]

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

## Model Summary Table

| Phase | Model / Procedure | Exact Specification Used | Purpose In The Project | Key Reason It Was Chosen |
|---|---|---|---|---|
| Phase 3 | Classical mean benchmark | `SARIMAX` on `nasdaq log_return` with order `(0,0,0)`, seasonal order `(0,0,0,0)`, constant term, and 8 lagged exogenous regressors | Forecast the conditional mean of returns and establish a disciplined linear benchmark | Interpretable, statistically grounded, and necessary as a baseline before claiming any value from more complex models |
| Phase 4 | Volatility benchmark | `GARCH(1,1)` with Student-t innovations on Phase 3 residuals | Forecast conditional variance / risk rather than direction | Financial returns exhibit volatility clustering and heavy tails; GARCH is the standard classical risk model |
| Phase 5 | Deep-learning benchmark | PatchTST-style transformer with 60-day lookback, 10-day patches, 5-day stride, 33 lagged multivariate features, and one-step-ahead NASDAQ return target | Test whether nonlinear temporal and cross-series structure improves forecasting beyond the linear benchmark | Inspired by the PatchTST paper and official repo, but adapted for local, explainable, reproducible execution |
| Phase 6 | Financial evaluation | 5-fold rolling walk-forward backtest with retraining, sign-based trading rule, 2 bps commissions, and 3 bps slippage | Evaluate whether forecasting gains translate into robust economic performance | Forecast accuracy alone is not enough in finance; models must be tested across regimes and under market frictions |

## Current Project State

- `Assignment1/start.ipynb` and `Assignment1/01_EDA.ipynb` were reviewed.
- Phase 1 preprocessing and stationarity code has been implemented and validated on the live dataset.
- Phase 2 STL decomposition has been implemented, exported, and interpreted on the live dataset.
- Phase 3 classical benchmarking has been implemented with `pmdarima` order selection, a SARIMAX fit, residual diagnostics, and updated visual analytics.
- Phase 4 volatility modeling has been implemented with a GARCH benchmark on the classical-model residuals.
- Phase 5 deep forecasting has been implemented with a PatchTST-style transformer benchmark and documented against the classical baseline.
- Phase 6 will add rolling-window backtesting, trading-rule simulation, and market-friction-aware financial evaluation.

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
