from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


@dataclass(frozen=True)
class PhaseOneConfig:
    """Configuration for the first project phase."""

    dataset_path: Path = Path("Assignment1/financial_regression.csv")
    output_dir: Path = Path("Assignment1/outputs/phase1")
    start_date: str = "2010-04-01"
    date_column: str = "date"
    macro_columns: tuple[str, ...] = ("GDP", "CPI", "us_rates_%")
    rsi_window: int = 14
    bollinger_window: int = 20
    bollinger_std: float = 2.0
    macd_fast_span: int = 12
    macd_slow_span: int = 26
    macd_signal_span: int = 9
    adf_regression: str = "c"
    adf_autolag: str = "AIC"


@dataclass(frozen=True)
class PhaseOneArtifacts:
    """Container returned by the phase-one pipeline."""

    cleaned_data: pd.DataFrame
    featured_data: pd.DataFrame
    modeling_data: pd.DataFrame
    adf_summary: pd.DataFrame
    rows_removed_before_start: int
    non_trading_rows_removed: int

    def summary(self) -> str:
        close_feature_count = len([col for col in self.featured_data.columns if col.endswith("log_return")])
        return (
            f"Cleaned rows: {len(self.cleaned_data)} | "
            f"Modeling rows: {len(self.modeling_data)} | "
            f"Log-return columns: {close_feature_count} | "
            f"ADF tests: {len(self.adf_summary)} | "
            f"Rows removed before {self.cleaned_data['date'].min().date()}: {self.rows_removed_before_start} | "
            f"Non-trading rows removed: {self.non_trading_rows_removed}"
        )


class FinancialDatasetLoader:
    """Load the mixed-frequency dataset and make alignment assumptions explicit."""

    def __init__(self, config: PhaseOneConfig) -> None:
        self.config = config

    def load_and_clean(self) -> tuple[pd.DataFrame, dict[str, int]]:
        raw = pd.read_csv(self.config.dataset_path, parse_dates=[self.config.date_column])
        self._validate_schema(raw)

        ordered = raw.sort_values(self.config.date_column).reset_index(drop=True)
        if ordered[self.config.date_column].duplicated().any():
            raise ValueError("Duplicate dates detected. A time-series index must be unique.")

        trimmed = ordered.loc[ordered[self.config.date_column] >= self.config.start_date].copy()
        rows_removed_before_start = len(ordered) - len(trimmed)

        close_columns = self._find_columns(trimmed, suffix="close")
        non_trading_mask = trimmed[close_columns].isna().all(axis=1)
        cleaned = trimmed.loc[~non_trading_mask].reset_index(drop=True)

        # Forward fill is appropriate for macro releases because each release
        # remains the latest known value until the next publication arrives.
        cleaned.loc[:, list(self.config.macro_columns)] = cleaned.loc[:, list(self.config.macro_columns)].ffill()

        metadata = {
            "rows_removed_before_start": rows_removed_before_start,
            "non_trading_rows_removed": int(non_trading_mask.sum()),
        }
        return cleaned, metadata

    def _validate_schema(self, frame: pd.DataFrame) -> None:
        required_columns = {self.config.date_column, *self.config.macro_columns}
        missing_columns = required_columns.difference(frame.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Dataset is missing required columns: {missing}")

    @staticmethod
    def _find_columns(frame: pd.DataFrame, suffix: str) -> list[str]:
        columns = [column for column in frame.columns if column.endswith(suffix)]
        if not columns:
            raise ValueError(f"No columns ending with '{suffix}' were found.")
        return columns


class TechnicalFeatureEngineer:
    """Create return-based and technical-analysis features from close prices."""

    def __init__(self, config: PhaseOneConfig) -> None:
        self.config = config

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        enriched = frame.copy()
        close_columns = self._find_close_columns(enriched)

        for close_column in close_columns:
            asset_name = close_column.removesuffix(" close")
            price_series = enriched[close_column]

            enriched[f"{asset_name} log_return"] = self._compute_log_returns(price_series)
            enriched[f"{asset_name} rsi_{self.config.rsi_window}"] = self._compute_rsi(
                price_series,
                window=self.config.rsi_window,
            )

            macd_frame = self._compute_macd(price_series)
            for macd_name, macd_values in macd_frame.items():
                enriched[f"{asset_name} {macd_name}"] = macd_values

            bollinger_frame = self._compute_bollinger_bands(price_series)
            for band_name, band_values in bollinger_frame.items():
                enriched[f"{asset_name} {band_name}"] = band_values

        return enriched

    @staticmethod
    def _find_close_columns(frame: pd.DataFrame) -> list[str]:
        return [column for column in frame.columns if column.endswith(" close")]

    @staticmethod
    def _compute_log_returns(price_series: pd.Series) -> pd.Series:
        return np.log(price_series / price_series.shift(1))

    @staticmethod
    def _compute_rsi(price_series: pd.Series, window: int) -> pd.Series:
        delta = price_series.diff()
        gains = delta.clip(lower=0.0)
        losses = -delta.clip(upper=0.0)

        # Wilder's exponential smoothing is the standard RSI formulation used by traders.
        avg_gain = gains.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
        avg_loss = losses.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
        relative_strength = avg_gain / avg_loss.replace(0.0, np.nan)
        return 100.0 - (100.0 / (1.0 + relative_strength))

    def _compute_macd(self, price_series: pd.Series) -> dict[str, pd.Series]:
        fast_ema = price_series.ewm(span=self.config.macd_fast_span, adjust=False).mean()
        slow_ema = price_series.ewm(span=self.config.macd_slow_span, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=self.config.macd_signal_span, adjust=False).mean()
        histogram = macd_line - signal_line

        return {
            "macd_line": macd_line,
            "macd_signal": signal_line,
            "macd_hist": histogram,
        }

    def _compute_bollinger_bands(self, price_series: pd.Series) -> dict[str, pd.Series]:
        rolling_mean = price_series.rolling(window=self.config.bollinger_window).mean()
        rolling_std = price_series.rolling(window=self.config.bollinger_window).std(ddof=0)

        upper_band = rolling_mean + (self.config.bollinger_std * rolling_std)
        lower_band = rolling_mean - (self.config.bollinger_std * rolling_std)
        zscore = (price_series - rolling_mean) / rolling_std.replace(0.0, np.nan)

        return {
            "bb_middle": rolling_mean,
            "bb_upper": upper_band,
            "bb_lower": lower_band,
            "bb_zscore": zscore,
        }


class StationarityAnalyzer:
    """Run Augmented Dickey-Fuller tests on levels and transformed series."""

    def __init__(self, config: PhaseOneConfig) -> None:
        self.config = config

    def analyze(self, frame: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        close_columns = [column for column in frame.columns if column.endswith(" close")]

        for close_column in close_columns:
            asset_name = close_column.removesuffix(" close")
            rows.append(self._run_adf(frame[close_column], asset_name, "price_level"))

            return_column = f"{asset_name} log_return"
            rows.append(self._run_adf(frame[return_column], asset_name, "log_return"))

        return pd.DataFrame(rows).sort_values(["asset", "transformation"]).reset_index(drop=True)

    def _run_adf(self, series: pd.Series, asset_name: str, transformation: str) -> dict[str, Any]:
        clean_series = series.dropna()

        if len(clean_series) < 30:
            return {
                "asset": asset_name,
                "transformation": transformation,
                "n_obs": len(clean_series),
                "adf_statistic": np.nan,
                "p_value": np.nan,
                "used_lag": np.nan,
                "critical_value_1pct": np.nan,
                "critical_value_5pct": np.nan,
                "critical_value_10pct": np.nan,
                "reject_unit_root_5pct": False,
                "status": "insufficient_observations",
            }

        try:
            statistic, p_value, used_lag, n_obs, critical_values, *_ = adfuller(
                clean_series,
                regression=self.config.adf_regression,
                autolag=self.config.adf_autolag,
            )
            return {
                "asset": asset_name,
                "transformation": transformation,
                "n_obs": int(n_obs),
                "adf_statistic": float(statistic),
                "p_value": float(p_value),
                "used_lag": int(used_lag),
                "critical_value_1pct": float(critical_values["1%"]),
                "critical_value_5pct": float(critical_values["5%"]),
                "critical_value_10pct": float(critical_values["10%"]),
                "reject_unit_root_5pct": bool(p_value < 0.05),
                "status": "ok",
            }
        except ValueError as error:
            return {
                "asset": asset_name,
                "transformation": transformation,
                "n_obs": len(clean_series),
                "adf_statistic": np.nan,
                "p_value": np.nan,
                "used_lag": np.nan,
                "critical_value_1pct": np.nan,
                "critical_value_5pct": np.nan,
                "critical_value_10pct": np.nan,
                "reject_unit_root_5pct": False,
                "status": f"error: {error}",
            }


class PhaseOnePipeline:
    """Orchestrate cleaning, feature engineering, and stationarity diagnostics."""

    def __init__(self, config: PhaseOneConfig | None = None) -> None:
        self.config = config or PhaseOneConfig()
        self.loader = FinancialDatasetLoader(self.config)
        self.feature_engineer = TechnicalFeatureEngineer(self.config)
        self.stationarity_analyzer = StationarityAnalyzer(self.config)

    def run(self, save_outputs: bool = True) -> PhaseOneArtifacts:
        cleaned_data, metadata = self.loader.load_and_clean()
        featured_data = self.feature_engineer.transform(cleaned_data)

        # Complete-case filtering is deferred until after feature creation so that
        # rolling indicators can consume the full history before warm-up rows are dropped.
        modeling_data = featured_data.dropna().reset_index(drop=True)
        adf_summary = self.stationarity_analyzer.analyze(featured_data)

        artifacts = PhaseOneArtifacts(
            cleaned_data=cleaned_data,
            featured_data=featured_data,
            modeling_data=modeling_data,
            adf_summary=adf_summary,
            rows_removed_before_start=metadata["rows_removed_before_start"],
            non_trading_rows_removed=metadata["non_trading_rows_removed"],
        )

        if save_outputs:
            self._save_outputs(artifacts)

        return artifacts

    def _save_outputs(self, artifacts: PhaseOneArtifacts) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        artifacts.cleaned_data.to_csv(self.config.output_dir / "cleaned_data.csv", index=False)
        artifacts.featured_data.to_csv(self.config.output_dir / "featured_data.csv", index=False)
        artifacts.modeling_data.to_csv(self.config.output_dir / "modeling_data.csv", index=False)
        artifacts.adf_summary.to_csv(self.config.output_dir / "adf_summary.csv", index=False)
