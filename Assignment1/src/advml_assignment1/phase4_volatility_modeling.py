from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera


@dataclass(frozen=True)
class PhaseFourConfig:
    """Configuration for GARCH volatility modeling on mean-model residuals."""

    phase3_train_path: Path = Path("Assignment1/outputs/phase3/train_fitted.csv")
    phase3_test_path: Path = Path("Assignment1/outputs/phase3/test_forecasts.csv")
    output_dir: Path = Path("Assignment1/outputs/phase4")
    residual_column_train: str = "residual"
    residual_column_test: str = "forecast_error"
    scale_factor: float = 100.0
    p: int = 1
    q: int = 1
    distribution: str = "t"
    arch_test_lags: int = 10
    ljung_box_lags: tuple[int, ...] = (5, 10, 20)


@dataclass(frozen=True)
class PhaseFourArtifacts:
    """Outputs produced by the GARCH phase."""

    combined_residuals: pd.DataFrame
    train_volatility: pd.DataFrame
    test_volatility_forecasts: pd.DataFrame
    parameter_summary: pd.DataFrame
    residual_diagnostics: pd.DataFrame
    qq_plot_data: pd.DataFrame
    model_metadata: dict[str, Any]

    def summary(self) -> str:
        persistence = self.model_metadata["persistence"]
        metrics = self.model_metadata["evaluation_metrics"]
        return (
            f"GARCH({self.model_metadata['order'][0]},{self.model_metadata['order'][1]}) | "
            f"Distribution: {self.model_metadata['distribution']} | "
            f"Persistence: {persistence:.4f} | "
            f"Train rows: {self.model_metadata['train_rows']} | "
            f"Test rows: {self.model_metadata['test_rows']} | "
            f"QLIKE: {metrics['qlike']:.6f}"
        )


class GarchVolatilityModeler:
    """Fit a GARCH model to Phase 3 residuals and forecast conditional risk."""

    def __init__(self, config: PhaseFourConfig) -> None:
        self.config = config

    def run(self) -> PhaseFourArtifacts:
        train_frame = pd.read_csv(self.config.phase3_train_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        test_frame = pd.read_csv(self.config.phase3_test_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

        combined_residuals = self._build_combined_residuals(train_frame, test_frame)
        train_rows = len(train_frame)
        scaled_residuals = combined_residuals["mean_residual"] * self.config.scale_factor

        model = arch_model(
            scaled_residuals,
            mean="Zero",
            vol="GARCH",
            p=self.config.p,
            q=self.config.q,
            dist=self.config.distribution,
            rescale=False,
        )
        result = model.fit(disp="off", last_obs=train_rows - 1)

        forecast = result.forecast(horizon=1, start=train_rows, reindex=True)

        train_volatility = self._build_train_volatility(train_frame, result, train_rows)
        test_volatility_forecasts = self._build_test_forecasts(test_frame, forecast, train_rows)
        parameter_summary = self._build_parameter_summary(result)
        residual_diagnostics = self._build_residual_diagnostics(train_volatility["standardized_residual"].dropna())
        qq_plot_data = self._build_qq_plot_data(train_volatility["standardized_residual"].dropna(), result.params["nu"])
        model_metadata = self._build_metadata(result, train_frame, test_frame, test_volatility_forecasts, residual_diagnostics)

        return PhaseFourArtifacts(
            combined_residuals=combined_residuals,
            train_volatility=train_volatility,
            test_volatility_forecasts=test_volatility_forecasts,
            parameter_summary=parameter_summary,
            residual_diagnostics=residual_diagnostics,
            qq_plot_data=qq_plot_data,
            model_metadata=model_metadata,
        )

    def _build_combined_residuals(self, train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> pd.DataFrame:
        return pd.concat(
            [
                train_frame[["date", self.config.residual_column_train]]
                .rename(columns={self.config.residual_column_train: "mean_residual"})
                .assign(segment="train"),
                test_frame[["date", self.config.residual_column_test]]
                .rename(columns={self.config.residual_column_test: "mean_residual"})
                .assign(segment="test"),
            ],
            ignore_index=True,
        )

    def _build_train_volatility(self, train_frame: pd.DataFrame, result, train_rows: int) -> pd.DataFrame:
        cond_vol = result.conditional_volatility.iloc[:train_rows] / self.config.scale_factor
        cond_var = cond_vol ** 2
        standardized = train_frame[self.config.residual_column_train] / cond_vol.replace(0.0, np.nan)
        return pd.DataFrame(
            {
                "date": train_frame["date"],
                "mean_residual": train_frame[self.config.residual_column_train],
                "conditional_volatility": cond_vol,
                "conditional_variance": cond_var,
                "standardized_residual": standardized,
                "squared_standardized_residual": standardized ** 2,
            }
        )

    def _build_test_forecasts(self, test_frame: pd.DataFrame, forecast, train_rows: int) -> pd.DataFrame:
        variance_pct = forecast.variance["h.1"].iloc[train_rows:].reset_index(drop=True)
        variance = variance_pct / (self.config.scale_factor ** 2)
        volatility = np.sqrt(variance)
        realized_residual = test_frame[self.config.residual_column_test].reset_index(drop=True)
        realized_sq_error = realized_residual ** 2

        output = pd.DataFrame(
            {
                "date": test_frame["date"],
                "mean_residual": realized_residual,
                "forecast_variance": variance,
                "forecast_volatility": volatility,
                "realized_sq_error": realized_sq_error,
            }
        )
        output["volatility_error"] = output["realized_sq_error"] - output["forecast_variance"]
        output["qlike_contribution"] = np.log(output["forecast_variance"]) + (
            output["realized_sq_error"] / output["forecast_variance"]
        )
        return output

    @staticmethod
    def _build_parameter_summary(result) -> pd.DataFrame:
        conf_int = result.conf_int()
        return pd.DataFrame(
            {
                "parameter": result.params.index,
                "estimate": result.params.values,
                "std_error": result.std_err.values,
                "t_stat": result.tvalues.values,
                "p_value": result.pvalues.values,
                "ci_lower": conf_int.iloc[:, 0].values,
                "ci_upper": conf_int.iloc[:, 1].values,
            }
        )

    def _build_residual_diagnostics(self, standardized_residuals: pd.Series) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []

        lb_std = acorr_ljungbox(standardized_residuals, lags=list(self.config.ljung_box_lags), return_df=True)
        for lag, row in lb_std.iterrows():
            rows.append(
                {
                    "test": "ljung_box_standardized_residual",
                    "lag": int(lag),
                    "statistic": float(row["lb_stat"]),
                    "p_value": float(row["lb_pvalue"]),
                    "extra_1": np.nan,
                    "extra_2": np.nan,
                }
            )

        lb_sq = acorr_ljungbox(standardized_residuals ** 2, lags=list(self.config.ljung_box_lags), return_df=True)
        for lag, row in lb_sq.iterrows():
            rows.append(
                {
                    "test": "ljung_box_squared_standardized_residual",
                    "lag": int(lag),
                    "statistic": float(row["lb_stat"]),
                    "p_value": float(row["lb_pvalue"]),
                    "extra_1": np.nan,
                    "extra_2": np.nan,
                }
            )

        arch_lm = het_arch(standardized_residuals, nlags=self.config.arch_test_lags)
        rows.append(
            {
                "test": "arch_lm",
                "lag": int(self.config.arch_test_lags),
                "statistic": float(arch_lm[0]),
                "p_value": float(arch_lm[1]),
                "extra_1": float(arch_lm[2]),
                "extra_2": float(arch_lm[3]),
            }
        )

        jb = jarque_bera(standardized_residuals)
        rows.append(
            {
                "test": "jarque_bera_standardized_residual",
                "lag": np.nan,
                "statistic": float(jb[0]),
                "p_value": float(jb[1]),
                "extra_1": float(jb[2]),  # skewness
                "extra_2": float(jb[3]),  # kurtosis
            }
        )

        return pd.DataFrame(rows)

    @staticmethod
    def _build_qq_plot_data(standardized_residuals: pd.Series, nu: float) -> pd.DataFrame:
        ordered = np.sort(np.asarray(standardized_residuals))
        probabilities = (np.arange(1, len(ordered) + 1) - 0.5) / len(ordered)
        theoretical = stats.t.ppf(probabilities, df=nu)
        reference = np.polyfit(theoretical, ordered, deg=1)
        reference_line = reference[0] * theoretical + reference[1]
        return pd.DataFrame(
            {
                "theoretical_quantile_t": theoretical,
                "sample_quantile": ordered,
                "reference_line": reference_line,
            }
        )

    def _build_metadata(
        self,
        result,
        train_frame: pd.DataFrame,
        test_frame: pd.DataFrame,
        test_volatility_forecasts: pd.DataFrame,
        residual_diagnostics: pd.DataFrame,
    ) -> dict[str, Any]:
        params = result.params
        alpha = float(params["alpha[1]"])
        beta = float(params["beta[1]"])
        omega = float(params["omega"])
        persistence = alpha + beta

        if persistence < 1.0:
            unconditional_variance_pct = omega / (1.0 - persistence)
            unconditional_volatility = math.sqrt(unconditional_variance_pct) / self.config.scale_factor
            half_life = math.log(0.5) / math.log(persistence) if persistence > 0 else 0.0
        else:
            unconditional_variance_pct = math.inf
            unconditional_volatility = math.inf
            half_life = math.inf

        variance_proxy = test_volatility_forecasts["realized_sq_error"]
        variance_forecast = test_volatility_forecasts["forecast_variance"]
        qlike = float(test_volatility_forecasts["qlike_contribution"].mean())
        variance_rmse = float(np.sqrt(np.mean((variance_proxy - variance_forecast) ** 2)))
        volatility_rmse = float(
            np.sqrt(
                np.mean(
                    (np.sqrt(variance_proxy) - test_volatility_forecasts["forecast_volatility"]) ** 2
                )
            )
        )

        diag_map: dict[str, dict[str, Any]] = {}
        for _, row in residual_diagnostics.iterrows():
            key = row["test"] if pd.isna(row["lag"]) else f"{row['test']}_lag_{int(row['lag'])}"
            diag_map[str(key)] = {
                "statistic": float(row["statistic"]),
                "p_value": float(row["p_value"]),
                "extra_1": None if pd.isna(row["extra_1"]) else float(row["extra_1"]),
                "extra_2": None if pd.isna(row["extra_2"]) else float(row["extra_2"]),
            }

        return {
            "order": [self.config.p, self.config.q],
            "distribution": self.config.distribution,
            "scale_factor": self.config.scale_factor,
            "train_rows": int(len(train_frame)),
            "test_rows": int(len(test_frame)),
            "train_start": str(train_frame["date"].min().date()),
            "train_end": str(train_frame["date"].max().date()),
            "test_start": str(test_frame["date"].min().date()),
            "test_end": str(test_frame["date"].max().date()),
            "omega": omega,
            "alpha_1": alpha,
            "beta_1": beta,
            "persistence": persistence,
            "nu": float(params["nu"]),
            "unconditional_volatility": unconditional_volatility,
            "half_life_periods": half_life,
            "loglikelihood": float(result.loglikelihood),
            "aic": float(result.aic),
            "bic": float(result.bic),
            "evaluation_metrics": {
                "qlike": qlike,
                "variance_rmse": variance_rmse,
                "volatility_rmse": volatility_rmse,
            },
            "diagnostics": diag_map,
        }


class PhaseFourPipeline:
    """Persist Phase 4 volatility outputs and summaries."""

    def __init__(self, config: PhaseFourConfig | None = None) -> None:
        self.config = config or PhaseFourConfig()
        self.modeler = GarchVolatilityModeler(self.config)

    def run(self, save_outputs: bool = True) -> PhaseFourArtifacts:
        artifacts = self.modeler.run()
        if save_outputs:
            self._save_outputs(artifacts)
        return artifacts

    def _save_outputs(self, artifacts: PhaseFourArtifacts) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        artifacts.combined_residuals.to_csv(self.config.output_dir / "combined_mean_residuals.csv", index=False)
        artifacts.train_volatility.to_csv(self.config.output_dir / "train_volatility.csv", index=False)
        artifacts.test_volatility_forecasts.to_csv(self.config.output_dir / "test_volatility_forecasts.csv", index=False)
        artifacts.parameter_summary.to_csv(self.config.output_dir / "garch_parameter_summary.csv", index=False)
        artifacts.residual_diagnostics.to_csv(self.config.output_dir / "garch_residual_diagnostics.csv", index=False)
        artifacts.qq_plot_data.to_csv(self.config.output_dir / "garch_qq_plot_data.csv", index=False)
        with open(self.config.output_dir / "garch_model_metadata.json", "w", encoding="utf-8") as handle:
            json.dump(artifacts.model_metadata, handle, indent=2)
