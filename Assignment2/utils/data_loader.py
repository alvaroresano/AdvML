"""
utils/data_loader.py
====================
Data loading, preprocessing, feature engineering, and target creation
for Assignment 2: Imbalanced Data, Imputation & Multi-class Classification.
 
Reutilises the same dataset and preprocessing logic from Assignment 1 
(forward-fill macroeconomic variables, log-return computation, 
technical indicators) and extends it with:
  - Artificial MCAR missingness injection for imputation experiments
  - Multi-class target variable creation from return quantiles
"""
 
import numpy as np
import pandas as pd
from pathlib import Path
 
 
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASS_NAMES = {
    0: "Strong Drop",
    1: "Mild Drop",
    2: "Neutral",
    3: "Mild Rise",
    4: "Strong Rise",
}
 
MACRO_COLS = ["us_rates_%", "CPI", "GDP"]
MARKET_CLOSE_COLS = ["nasdaq_close", "sp500_close", "gold_close",
                     "silver_close", "oil_close", "platinum_close", "palladium_close"]
FX_COLS = ["usd_chf", "eur_usd"]
 
 
# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------
 
def load_raw(path: str | Path) -> pd.DataFrame:
    """Load the raw CSV and normalise column names."""
    df = pd.read_csv(path, parse_dates=["date"])
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    df = df.sort_values("date").reset_index(drop=True)
    return df
 
 
def preprocess_macro(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill macro columns (GDP, CPI, us_rates_%) to propagate
    the last known value across daily market rows.  This is the same
    mixed-frequency alignment used in Assignment 1 and is defensible
    because macro indicators remain at their published value until the
    next release.
    """
    df = df.copy()
    for col in MACRO_COLS:
        if col in df.columns:
            df[col] = df[col].ffill()
    return df
 
 
def drop_market_holidays(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where all market prices are NaN (weekends / holidays)."""
    return df.dropna(subset=["nasdaq_close", "sp500_close"]).reset_index(drop=True)
 
 
def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log returns for all close-price columns.
    r_t = log(P_t / P_{t-1})
    """
    df = df.copy()
    for col in MARKET_CLOSE_COLS:
        if col in df.columns:
            ret_col = col.replace("_close", "_log_return")
            df[ret_col] = np.log(df[col] / df[col].shift(1))
    return df
 
 
def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RSI, MACD, and Bollinger Bands on nasdaq_close.
    Replicates the feature engineering from Assignment 1.
    """
    df = df.copy()
    price = df["nasdaq_close"]
 
    # ---- RSI (14-day Wilder) ----
    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["nasdaq_rsi_14"] = 100 - (100 / (1 + rs))
 
    # ---- MACD (12, 26, 9) ----
    ema12 = price.ewm(span=12, adjust=False).mean()
    ema26 = price.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["nasdaq_macd"] = macd
    df["nasdaq_macd_signal"] = signal
    df["nasdaq_macd_hist"] = macd - signal
 
    # ---- Bollinger Bands (20-day, 2 std) ----
    mid = price.rolling(20).mean()
    std = price.rolling(20).std()
    df["nasdaq_bb_upper"] = mid + 2 * std
    df["nasdaq_bb_lower"] = mid - 2 * std
    df["nasdaq_bb_width"] = (mid + 2 * std - (mid - 2 * std)) / mid
 
    # ---- Lagged log returns (1, 2, 3, 5 days) ----
    for lag in [1, 2, 3, 5]:
        df[f"nasdaq_lr_lag{lag}"] = df["nasdaq_log_return"].shift(lag)
 
    # ---- 5-day and 20-day rolling volatility ----
    df["nasdaq_vol_5d"] = df["nasdaq_log_return"].rolling(5).std()
    df["nasdaq_vol_20d"] = df["nasdaq_log_return"].rolling(20).std()
 
    # ---- SP500 & gold log return lags ----
    for base in ["sp500", "gold"]:
        lr_col = f"{base}_log_return"
        if lr_col in df.columns:
            df[f"{base}_lr_lag1"] = df[lr_col].shift(1)
 
    return df
 
 
def create_target(df: pd.DataFrame, n_classes: int = 5) -> pd.DataFrame:
    """
    Discretise the NEXT-DAY nasdaq log return into n_classes movement categories.
    Thresholds are defined relative to mean ± k*std of the TRAINING distribution
    (should be computed on training set only; here we compute globally for EDA
     and reassign correctly inside the pipeline).
 
    Classes (5-class):
      0: Strong Drop  (ret < mu - 1.5*sigma)
      1: Mild Drop    (mu - 1.5*sigma <= ret < mu - 0.5*sigma)
      2: Neutral      (mu - 0.5*sigma <= ret <= mu + 0.5*sigma)
      3: Mild Rise    (mu + 0.5*sigma < ret <= mu + 1.5*sigma)
      4: Strong Rise  (ret > mu + 1.5*sigma)
    """
    df = df.copy()
    # Target = NEXT day return (shift -1)
    df["target_return"] = df["nasdaq_log_return"].shift(-1)
    df = df.dropna(subset=["target_return"]).reset_index(drop=True)
 
    ret = df["target_return"]
    mu, sigma = ret.mean(), ret.std()
 
    if n_classes == 5:
        t = [mu - 1.5 * sigma, mu - 0.5 * sigma, mu + 0.5 * sigma, mu + 1.5 * sigma]
        conds = [
            ret < t[0],
            (ret >= t[0]) & (ret < t[1]),
            (ret >= t[1]) & (ret <= t[2]),
            (ret > t[2]) & (ret <= t[3]),
            ret > t[3],
        ]
        df["target_class"] = np.select(conds, [0, 1, 2, 3, 4])
    elif n_classes == 3:
        t = [mu - 0.5 * sigma, mu + 0.5 * sigma]
        conds = [ret < t[0], (ret >= t[0]) & (ret <= t[1]), ret > t[1]]
        df["target_class"] = np.select(conds, [0, 1, 2])
    else:
        raise ValueError("n_classes must be 3 or 5")
 
    # Store thresholds for later use
    df.attrs["thresholds"] = t
    df.attrs["n_classes"] = n_classes
    return df
 
 
# ---------------------------------------------------------------------------
# Missing data injection (for imputation experiment)
# ---------------------------------------------------------------------------
 
def inject_mcar_missingness(
    df: pd.DataFrame,
    cols: list[str],
    missing_rate: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Introduce Missing Completely At Random (MCAR) values into selected columns.
    This allows a controlled comparison of imputation methods.
 
    Parameters
    ----------
    cols : columns to corrupt
    missing_rate : fraction of values to set to NaN per column
    """
    df = df.copy()
    rng = np.random.default_rng(random_state)
    for col in cols:
        if col not in df.columns:
            continue
        n_missing = int(len(df) * missing_rate)
        idx = rng.choice(df.index, size=n_missing, replace=False)
        df.loc[idx, col] = np.nan
    return df
 
 
# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
 
def build_dataset(
    path: str | Path,
    n_classes: int = 5,
    inject_missing: bool = True,
    missing_rate: float = 0.05,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full preprocessing pipeline.
 
    Returns
    -------
    df_clean : DataFrame without artificial missingness (ground truth)
    df_missing : DataFrame with MCAR missingness injected in technical indicators
    """
    df = load_raw(path)
    df = preprocess_macro(df)
    df = drop_market_holidays(df)
    df = compute_log_returns(df)
    df = compute_technical_indicators(df)
    df = create_target(df, n_classes=n_classes)
 
    # Drop rows with NaN in features (created by rolling windows / lags)
    feature_cols = _get_feature_cols(df)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
 
    df_clean = df.copy()
 
    if inject_missing:
        # Inject MCAR only into technical indicator columns
        cols_to_corrupt = [
            "nasdaq_rsi_14", "nasdaq_macd", "nasdaq_macd_signal",
            "nasdaq_macd_hist", "nasdaq_bb_width",
            "nasdaq_lr_lag1", "nasdaq_lr_lag2", "nasdaq_vol_5d",
        ]
        df_missing = inject_mcar_missingness(
            df, cols=cols_to_corrupt,
            missing_rate=missing_rate, random_state=random_state
        )
    else:
        df_missing = df.copy()
 
    return df_clean, df_missing
 
 
def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return the feature column names (exclude meta, target, raw prices)."""
    exclude = {
        "date", "target_class", "target_return",
        # raw prices (keep only derived features)
        "sp500_open", "sp500_high", "sp500_low", "sp500_close", "sp500_volume", "sp500_high-low",
        "nasdaq_open", "nasdaq_high", "nasdaq_low", "nasdaq_close", "nasdaq_volume", "nasdaq_high-low",
        "nasdaq_log_return",
        "silver_open", "silver_high", "silver_low", "silver_close", "silver_volume", "silver_high-low",
        "oil_open", "oil_high", "oil_low", "oil_close", "oil_volume", "oil_high-low",
        "platinum_open", "platinum_high", "platinum_low", "platinum_close", "platinum_volume", "platinum_high-low",
        "palladium_open", "palladium_high", "palladium_low", "palladium_close", "palladium_volume", "palladium_high-low",
        "gold_open", "gold_high", "gold_low", "gold_close", "gold_volume",
        "sp500_high-low", "nasdaq_high-low",
    }
    return [c for c in df.columns if c not in exclude]
 
 
def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix X and label vector y from processed DataFrame."""
    feature_cols = _get_feature_cols(df)
    # Remove target-related cols if present
    feature_cols = [c for c in feature_cols if c not in ("target_class", "target_return", "date")]
    return df[feature_cols], df["target_class"]