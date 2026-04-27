# Notes from the underground

## Prices vs Returns vs Log Returns

The first modeling decision in financial time series is whether to work on prices or on transformed series. Raw prices are usually non-stationary: their mean and variance structure drift (change) over time and generally exhibit exponential growth trajectories over long horizons, which breaks the assumptions behind ARIMA-class models and weakens statistical inference. Therefore, that's why we choose returns.

Simple returns measure the straightforward percentage change between two periods and are calculated as the difference between the current and previous price divided by the previous price. While intuitive and asset-additive—meaning the return of a portfolio is the weighted sum of the simple returns of its constituent assets—they present significant statistical drawbacks for time series modeling. Simple returns are bounded below by -100%, as an asset's price cannot drop below zero, but they are theoretically unbounded on the upside, leading to an asymmetric distribution. Furthermore, the product of normally distributed variables does not yield a normal distribution, complicating probabilistic modeling.

### Why log returns instead of simple returns

Logarithmic returns, calculated as the natural logarithm of the ratio of the current price to the previous price, are the standard for advanced time series forecasting. They offer several mathematical advantages. First, log returns compound additively over time, meaning the cumulative log return over a sequence of periods is simply the sum of the individual period log returns. Second, they possess perfect symmetry. If an asset's price increases from a baseline to a higher value and then returns to the baseline, the positive and negative log returns are absolute equals, unlike simple returns which mathematically distort the recovery required from a drawdown. Finally, log returns map the domain of positive real numbers to the entire real line, aligning perfectly with the assumptions of standard regression models and Gaussian error distributions. If asset prices follow a geometric Brownian motion, their log returns are normally distributed

STL means Seasonal-Trend decomposition using Loess. The idea is simple: a time series is often a mixture of three things:
    1. A slow-moving long-term direction, the trend
    2. A repeating pattern, the seasonality
    3. Everything left over, the residual or shock

For finance, one subtle point matters a lot: prices often behave more like a multiplicative process than an additive one. In plain words, a 2% move when oil is at 100 is not the same size in dollars as a 2% move when oil is at 40. That is why we used log returns, not raw prices.

## STL decomposition
