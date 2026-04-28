from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


@dataclass(frozen=True)
class PhaseTwoConfig:
    """Configuration for STL decomposition of trading-time financial series."""

    input_data_path: Path = Path("Assignment1/outputs/phase1/cleaned_data.csv")
    output_dir: Path = Path("Assignment1/outputs/phase2")
    date_column: str = "date"
    stl_period: int = 5
    robust: bool = True
    plot_format: str = "png"


@dataclass(frozen=True)
class PhaseTwoArtifacts:
    """Artifacts produced by the STL decomposition phase."""

    decomposition_components: pd.DataFrame
    decomposition_summary: pd.DataFrame
    analyzed_assets: tuple[str, ...]

    def summary(self) -> str:
        return (
            f"Assets decomposed: {len(self.analyzed_assets)} | "
            f"STL period: trading-week ({self.decomposition_summary['stl_period'].iloc[0]}) | "
            f"Average trend strength: {self.decomposition_summary['trend_strength'].mean():.4f} | "
            f"Average seasonal strength: {self.decomposition_summary['seasonal_strength'].mean():.4f}"
        )


class STLDecomposer:
    """Apply STL to log-price series so multiplicative structure becomes additive."""

    def __init__(self, config: PhaseTwoConfig) -> None:
        self.config = config

    def run(self) -> PhaseTwoArtifacts:
        frame = pd.read_csv(self.config.input_data_path, parse_dates=[self.config.date_column])
        close_columns = [column for column in frame.columns if column.endswith(" close")]
        if not close_columns:
            raise ValueError("No close-price columns were found in the Phase 2 input data.")

        components: list[pd.DataFrame] = []
        summary_rows: list[dict[str, float | str | int]] = []

        plots_dir = self.config.output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        for close_column in close_columns:
            asset_name = close_column.removesuffix(" close")
            asset_components, summary_row = self._decompose_asset(
                dates=frame[self.config.date_column],
                close_series=frame[close_column],
                asset_name=asset_name,
            )
            components.append(asset_components)
            summary_rows.append(summary_row)
            self._save_plot(asset_components, asset_name, plots_dir)

        components_frame = pd.concat(components, ignore_index=True)
        summary_frame = pd.DataFrame(summary_rows).sort_values("asset").reset_index(drop=True)

        return PhaseTwoArtifacts(
            decomposition_components=components_frame,
            decomposition_summary=summary_frame,
            analyzed_assets=tuple(summary_frame["asset"]),
        )

    def _decompose_asset(
        self,
        dates: pd.Series,
        close_series: pd.Series,
        asset_name: str,
    ) -> tuple[pd.DataFrame, dict[str, float | str | int]]:
        if (close_series <= 0).any():
            raise ValueError(f"STL on log prices requires strictly positive values. Invalid values found for {asset_name}.")

        log_price = np.log(close_series)
        stl_result = STL(log_price, period=self.config.stl_period, robust=self.config.robust).fit()

        components = pd.DataFrame(
            {
                "date": dates,
                "asset": asset_name,
                "observed_log_price": log_price,
                "trend": stl_result.trend,
                "seasonal": stl_result.seasonal,
                "residual": stl_result.resid,
            }
        )

        summary = {
            "asset": asset_name,
            "n_obs": int(len(components)),
            "stl_period": int(self.config.stl_period),
            "trend_strength": self._trend_strength(components),
            "seasonal_strength": self._seasonal_strength(components),
            "seasonal_amplitude": float(components["seasonal"].max() - components["seasonal"].min()),
            "residual_std": float(components["residual"].std(ddof=0)),
            "trend_std": float(components["trend"].std(ddof=0)),
            "residual_share_of_variance": float(
                components["residual"].var(ddof=0) / components["observed_log_price"].var(ddof=0)
            ),
        }

        return components, summary

    @staticmethod
    def _trend_strength(components: pd.DataFrame) -> float:
        resid_var = components["residual"].var(ddof=0)
        trend_plus_resid_var = (components["trend"] + components["residual"]).var(ddof=0)
        if trend_plus_resid_var == 0:
            return 0.0
        return float(max(0.0, 1.0 - resid_var / trend_plus_resid_var))

    @staticmethod
    def _seasonal_strength(components: pd.DataFrame) -> float:
        resid_var = components["residual"].var(ddof=0)
        seasonal_plus_resid_var = (components["seasonal"] + components["residual"]).var(ddof=0)
        if seasonal_plus_resid_var == 0:
            return 0.0
        return float(max(0.0, 1.0 - resid_var / seasonal_plus_resid_var))

    def _save_plot(self, components: pd.DataFrame, asset_name: str, plots_dir: Path) -> None:
        figure, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

        series_map = {
            "Observed Log Price": components["observed_log_price"],
            "Trend": components["trend"],
            "Seasonal": components["seasonal"],
            "Residual": components["residual"],
        }

        for axis, (title, series) in zip(axes, series_map.items()):
            axis.plot(components["date"], series, linewidth=1.0)
            axis.set_title(f"{asset_name}: {title}")
            axis.grid(alpha=0.2)

        figure.tight_layout()
        figure.savefig(plots_dir / f"{asset_name}_stl.{self.config.plot_format}", dpi=150, bbox_inches="tight")
        plt.close(figure)


class PhaseTwoPipeline:
    """Run STL decomposition and persist summary tables for reporting."""

    def __init__(self, config: PhaseTwoConfig | None = None) -> None:
        self.config = config or PhaseTwoConfig()
        self.decomposer = STLDecomposer(self.config)

    def run(self, save_outputs: bool = True) -> PhaseTwoArtifacts:
        artifacts = self.decomposer.run()
        if save_outputs:
            self._save_outputs(artifacts)
        return artifacts

    def _save_outputs(self, artifacts: PhaseTwoArtifacts) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        artifacts.decomposition_components.to_csv(
            self.config.output_dir / "stl_decomposition_components.csv",
            index=False,
        )
        artifacts.decomposition_summary.to_csv(
            self.config.output_dir / "stl_decomposition_summary.csv",
            index=False,
        )
