from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .phase5_deep_forecasting import PatchTSTForecaster


@dataclass(frozen=True)
class PhaseSixConfig:
    """Configuration for rolling-window backtesting of the Phase 3 and Phase 5 models."""

    phase3_design_path: Path = Path("Assignment1/outputs/phase3/phase3_design_data.csv")
    phase5_design_path: Path = Path("Assignment1/outputs/phase5/phase5_design_data.csv")
    output_dir: Path = Path("Assignment1/outputs/phase6")
    train_window: int = 2000
    validation_window: int = 252
    test_window: int = 252
    step_size: int = 252
    lookback_window: int = 60
    commission_bps: float = 2.0
    slippage_bps: float = 3.0
    annualization_factor: int = 252
    phase5_patch_length: int = 10
    phase5_patch_stride: int = 5
    phase5_d_model: int = 32
    phase5_num_heads: int = 4
    phase5_num_layers: int = 2
    phase5_feedforward_dim: int = 64
    phase5_dropout: float = 0.10
    phase5_learning_rate: float = 1e-3
    phase5_weight_decay: float = 1e-4
    phase5_batch_size: int = 64
    phase5_max_epochs: int = 10
    phase5_early_stopping_patience: int = 4
    phase5_gradient_clip_norm: float = 1.0
    random_seed: int = 7
    device: str = "cpu"
    auto_arima_max_p: int = 5
    auto_arima_max_q: int = 5
    auto_arima_max_d: int = 1


@dataclass(frozen=True)
class PhaseSixArtifacts:
    """Outputs produced by the rolling backtesting phase."""

    fold_definitions: pd.DataFrame
    prediction_records: pd.DataFrame
    fold_metrics: pd.DataFrame
    strategy_daily_returns: pd.DataFrame
    strategy_summary: pd.DataFrame
    model_metadata: dict[str, Any]

    def summary(self) -> str:
        summary = self.strategy_summary.sort_values(["model", "metric"]).reset_index(drop=True)
        patchtst_sharpe = summary.loc[
            (summary["model"] == "phase5_patchtst") & (summary["metric"] == "annualized_sharpe_ratio"),
            "value",
        ]
        sarimax_sharpe = summary.loc[
            (summary["model"] == "phase3_sarimax") & (summary["metric"] == "annualized_sharpe_ratio"),
            "value",
        ]
        return (
            f"Folds: {self.model_metadata['num_folds']} | "
            f"Commission: {self.model_metadata['commission_bps']} bps | "
            f"Slippage: {self.model_metadata['slippage_bps']} bps | "
            f"SARIMAX Sharpe: {float(sarimax_sharpe.iloc[0]):.4f} | "
            f"PatchTST Sharpe: {float(patchtst_sharpe.iloc[0]):.4f}"
        )


class RollingWindowBacktester:
    """Run rolling-window forecast evaluation and cost-aware trading simulation."""

    def __init__(self, config: PhaseSixConfig) -> None:
        self.config = config

    def run(self) -> PhaseSixArtifacts:
        self._set_random_seed()

        phase3_design = pd.read_csv(self.config.phase3_design_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        phase5_design = pd.read_csv(self.config.phase5_design_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

        if not phase3_design["date"].equals(phase5_design["date"]):
            raise ValueError("Phase 3 and Phase 5 design matrices are not aligned on the same dates.")

        folds = self._build_folds(len(phase3_design))

        prediction_frames: list[pd.DataFrame] = []
        fold_metrics_rows: list[dict[str, Any]] = []

        for fold_id, fold in enumerate(folds, start=1):
            sarimax_predictions, sarimax_meta = self._run_phase3_fold(phase3_design, fold, fold_id)
            patchtst_predictions, patchtst_meta = self._run_phase5_fold(phase5_design, fold, fold_id)

            prediction_frames.extend([sarimax_predictions, patchtst_predictions])
            fold_metrics_rows.extend(
                [
                    self._compute_fold_metrics(sarimax_predictions, sarimax_meta),
                    self._compute_fold_metrics(patchtst_predictions, patchtst_meta),
                ]
            )

        prediction_records = pd.concat(prediction_frames, ignore_index=True).sort_values(["model", "date"]).reset_index(drop=True)
        strategy_daily_returns = self._build_strategy_returns(prediction_records)
        strategy_summary = self._build_strategy_summary(strategy_daily_returns)
        fold_definitions = pd.DataFrame(folds)
        model_metadata = self._build_metadata(folds, fold_metrics_rows, strategy_summary)

        return PhaseSixArtifacts(
            fold_definitions=fold_definitions,
            prediction_records=prediction_records,
            fold_metrics=pd.DataFrame(fold_metrics_rows),
            strategy_daily_returns=strategy_daily_returns,
            strategy_summary=strategy_summary,
            model_metadata=model_metadata,
        )

    def _set_random_seed(self) -> None:
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 4)))

    def _build_folds(self, n_obs: int) -> list[dict[str, Any]]:
        folds: list[dict[str, Any]] = []
        fold_id = 1
        train_start = 0

        while True:
            train_end = train_start + self.config.train_window
            validation_end = train_end + self.config.validation_window
            test_end = validation_end + self.config.test_window
            if test_end > n_obs:
                break

            folds.append(
                {
                    "fold_id": fold_id,
                    "train_start": train_start,
                    "train_end": train_end,
                    "validation_start": train_end,
                    "validation_end": validation_end,
                    "test_start": validation_end,
                    "test_end": test_end,
                }
            )
            fold_id += 1
            train_start += self.config.step_size

        if not folds:
            raise ValueError("No folds could be created with the current Phase 6 configuration.")
        return folds

    def _run_phase3_fold(
        self,
        design_data: pd.DataFrame,
        fold: dict[str, Any],
        fold_id: int,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        feature_columns = [column for column in design_data.columns if column not in {"date", "target"}]
        train_data = design_data.iloc[fold["train_start"] : fold["train_end"]].reset_index(drop=True)
        test_data = design_data.iloc[fold["test_start"] : fold["test_end"]].reset_index(drop=True)

        x_train = train_data[feature_columns]
        x_test = test_data[feature_columns]
        means = x_train.mean()
        stds = x_train.std(ddof=0).replace(0.0, 1.0)
        x_train_scaled = (x_train - means) / stds
        x_test_scaled = (x_test - means) / stds

        auto_model = auto_arima(
            y=train_data["target"],
            X=x_train_scaled,
            seasonal=False,
            m=1,
            information_criterion="aic",
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            trace=False,
            max_p=self.config.auto_arima_max_p,
            max_q=self.config.auto_arima_max_q,
            max_d=self.config.auto_arima_max_d,
            start_p=0,
            start_q=0,
            d=None,
            with_intercept=True,
        )
        trend = "c" if getattr(auto_model, "with_intercept", True) else "n"
        model = SARIMAX(
            endog=train_data["target"],
            exog=x_train_scaled,
            order=auto_model.order,
            seasonal_order=auto_model.seasonal_order,
            trend=trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)

        forecast_result = model.get_forecast(steps=len(test_data), exog=x_test_scaled)
        forecast_frame = forecast_result.summary_frame(alpha=0.05)
        predictions = pd.DataFrame(
            {
                "model": "phase3_sarimax",
                "fold_id": fold_id,
                "date": test_data["date"],
                "actual": test_data["target"],
                "forecast": forecast_frame["mean"].to_numpy(),
            }
        )
        predictions["forecast_error"] = predictions["actual"] - predictions["forecast"]
        predictions["predicted_direction"] = np.sign(predictions["forecast"])
        predictions["actual_direction"] = np.sign(predictions["actual"])
        predictions["direction_correct"] = predictions["predicted_direction"] == predictions["actual_direction"]

        metadata = {
            "model": "phase3_sarimax",
            "fold_id": fold_id,
            "order": list(auto_model.order),
            "seasonal_order": list(auto_model.seasonal_order),
        }
        return predictions, metadata

    def _run_phase5_fold(
        self,
        design_data: pd.DataFrame,
        fold: dict[str, Any],
        fold_id: int,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        feature_columns = [column for column in design_data.columns if column not in {"date", "target"}]
        feature_values = design_data[feature_columns].to_numpy(dtype=np.float32)
        target_values = design_data["target"].to_numpy(dtype=np.float32)
        dates = design_data["date"].to_numpy()

        train_slice = slice(fold["train_start"], fold["train_end"])
        feature_mean = feature_values[train_slice].mean(axis=0)
        feature_std = feature_values[train_slice].std(axis=0)
        feature_std = np.where(feature_std == 0.0, 1.0, feature_std)

        target_mean = float(target_values[train_slice].mean())
        target_std = float(target_values[train_slice].std())
        target_std = 1.0 if target_std == 0.0 else target_std

        scaled_features = (feature_values - feature_mean) / feature_std
        scaled_target = (target_values - target_mean) / target_std

        train_indices = np.arange(fold["train_start"] + self.config.lookback_window, fold["train_end"], dtype=int)
        validation_indices = np.arange(fold["validation_start"], fold["validation_end"], dtype=int)
        test_indices = np.arange(fold["test_start"], fold["test_end"], dtype=int)

        train_windows = self._build_windows(scaled_features, scaled_target, dates, train_indices)
        validation_windows = self._build_windows(scaled_features, scaled_target, dates, validation_indices)
        test_windows = self._build_windows(scaled_features, scaled_target, dates, test_indices)

        model = PatchTSTForecaster(
            num_channels=len(feature_columns),
            lookback_window=self.config.lookback_window,
            patch_length=self.config.phase5_patch_length,
            patch_stride=self.config.phase5_patch_stride,
            d_model=self.config.phase5_d_model,
            num_heads=self.config.phase5_num_heads,
            num_layers=self.config.phase5_num_layers,
            feedforward_dim=self.config.phase5_feedforward_dim,
            dropout=self.config.phase5_dropout,
        ).to(self.config.device)

        training_history, best_state, best_epoch = self._train_patchtst(model, train_windows, validation_windows)
        model.load_state_dict(best_state)

        test_predictions = self._predict_patchtst(model, test_windows, target_mean, target_std)
        test_predictions.insert(0, "model", "phase5_patchtst")
        test_predictions.insert(1, "fold_id", fold_id)

        metadata = {
            "model": "phase5_patchtst",
            "fold_id": fold_id,
            "best_epoch": best_epoch,
            "train_windows": int(len(train_indices)),
            "validation_windows": int(len(validation_indices)),
            "history_tail": training_history.tail(3).to_dict(orient="records"),
        }
        return test_predictions, metadata

    def _build_windows(
        self,
        feature_matrix: np.ndarray,
        target_vector: np.ndarray,
        dates: np.ndarray,
        target_indices: np.ndarray,
    ) -> dict[str, Any]:
        windows = np.stack(
            [feature_matrix[index - self.config.lookback_window : index] for index in target_indices],
            axis=0,
        )
        return {
            "inputs": windows.astype(np.float32),
            "targets": target_vector[target_indices].astype(np.float32),
            "dates": pd.to_datetime(dates[target_indices]),
        }

    def _train_patchtst(
        self,
        model: PatchTSTForecaster,
        train_windows: dict[str, Any],
        validation_windows: dict[str, Any],
    ) -> tuple[pd.DataFrame, dict[str, torch.Tensor], int]:
        train_dataset = TensorDataset(
            torch.tensor(train_windows["inputs"], dtype=torch.float32),
            torch.tensor(train_windows["targets"], dtype=torch.float32),
        )
        validation_dataset = TensorDataset(
            torch.tensor(validation_windows["inputs"], dtype=torch.float32),
            torch.tensor(validation_windows["targets"], dtype=torch.float32),
        )
        train_loader = DataLoader(train_dataset, batch_size=self.config.phase5_batch_size, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=self.config.phase5_batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.phase5_learning_rate,
            weight_decay=self.config.phase5_weight_decay,
        )
        loss_fn = nn.MSELoss()

        best_state: dict[str, torch.Tensor] | None = None
        best_validation_loss = math.inf
        best_epoch = -1
        patience_counter = 0
        history_rows: list[dict[str, float | int]] = []

        for epoch in range(1, self.config.phase5_max_epochs + 1):
            model.train()
            train_losses: list[float] = []
            for features, target in train_loader:
                features = features.to(self.config.device)
                target = target.to(self.config.device)

                optimizer.zero_grad()
                prediction = model(features)
                loss = loss_fn(prediction, target)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.config.phase5_gradient_clip_norm)
                optimizer.step()
                train_losses.append(float(loss.item()))

            model.eval()
            validation_losses: list[float] = []
            with torch.no_grad():
                for features, target in validation_loader:
                    features = features.to(self.config.device)
                    target = target.to(self.config.device)
                    prediction = model(features)
                    validation_losses.append(float(loss_fn(prediction, target).item()))

            train_loss = float(np.mean(train_losses))
            validation_loss = float(np.mean(validation_losses))
            history_rows.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "train_rmse": math.sqrt(train_loss),
                    "validation_rmse": math.sqrt(validation_loss),
                }
            )

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_epoch = epoch
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.phase5_early_stopping_patience:
                    break

        if best_state is None:
            raise RuntimeError("PatchTST fold training did not produce a valid checkpoint.")

        return pd.DataFrame(history_rows), best_state, best_epoch

    def _predict_patchtst(
        self,
        model: PatchTSTForecaster,
        test_windows: dict[str, Any],
        target_mean: float,
        target_std: float,
    ) -> pd.DataFrame:
        model.eval()
        inputs = torch.tensor(test_windows["inputs"], dtype=torch.float32, device=self.config.device)

        with torch.no_grad():
            prediction_scaled = model(inputs).detach().cpu().numpy()

        actual = test_windows["targets"] * target_std + target_mean
        forecast = prediction_scaled * target_std + target_mean
        predictions = pd.DataFrame(
            {
                "date": test_windows["dates"],
                "actual": actual,
                "forecast": forecast,
            }
        )
        predictions["forecast_error"] = predictions["actual"] - predictions["forecast"]
        predictions["predicted_direction"] = np.sign(predictions["forecast"])
        predictions["actual_direction"] = np.sign(predictions["actual"])
        predictions["direction_correct"] = predictions["predicted_direction"] == predictions["actual_direction"]
        return predictions

    def _compute_fold_metrics(self, predictions: pd.DataFrame, metadata: dict[str, Any]) -> dict[str, Any]:
        actual = predictions["actual"].to_numpy()
        forecast = predictions["forecast"].to_numpy()
        return {
            **metadata,
            "rmse": float(np.sqrt(mean_squared_error(actual, forecast))),
            "mae": float(mean_absolute_error(actual, forecast)),
            "directional_accuracy": float(predictions["direction_correct"].mean()),
            "mean_forecast_error": float(predictions["forecast_error"].mean()),
        }

    def _build_strategy_returns(self, prediction_records: pd.DataFrame) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        cost_rate = (self.config.commission_bps + self.config.slippage_bps) / 10000.0

        for model_name, frame in prediction_records.groupby("model", sort=False):
            ordered = frame.sort_values("date").reset_index(drop=True).copy()
            ordered["position"] = np.sign(ordered["forecast"]).astype(float)
            ordered["previous_position"] = ordered["position"].shift(1).fillna(0.0)
            ordered["turnover"] = (ordered["position"] - ordered["previous_position"]).abs()
            ordered["gross_return"] = ordered["position"] * ordered["actual"]
            ordered["transaction_cost"] = ordered["turnover"] * cost_rate
            ordered["net_return"] = ordered["gross_return"] - ordered["transaction_cost"]
            ordered["cumulative_gross"] = (1.0 + ordered["gross_return"]).cumprod()
            ordered["cumulative_net"] = (1.0 + ordered["net_return"]).cumprod()
            ordered["model"] = model_name
            frames.append(ordered)

        return pd.concat(frames, ignore_index=True)

    def _build_strategy_summary(self, strategy_daily_returns: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for model_name, frame in strategy_daily_returns.groupby("model", sort=False):
            net_returns = frame["net_return"].to_numpy()
            cumulative_net = frame["cumulative_net"].to_numpy()
            running_peak = np.maximum.accumulate(cumulative_net)
            drawdown = (cumulative_net / running_peak) - 1.0
            sharpe = (
                math.sqrt(self.config.annualization_factor) * net_returns.mean() / net_returns.std(ddof=0)
                if net_returns.std(ddof=0) > 0
                else 0.0
            )
            metrics = {
                "rmse": float(np.sqrt(mean_squared_error(frame["actual"], frame["forecast"]))),
                "mae": float(mean_absolute_error(frame["actual"], frame["forecast"])),
                "directional_accuracy": float(frame["direction_correct"].mean()),
                "average_turnover": float(frame["turnover"].mean()),
                "gross_cumulative_return": float(frame["cumulative_gross"].iloc[-1] - 1.0),
                "net_cumulative_return": float(frame["cumulative_net"].iloc[-1] - 1.0),
                "annualized_sharpe_ratio": float(sharpe),
                "maximum_drawdown": float(drawdown.min()),
            }
            rows.extend(
                [{"model": model_name, "metric": metric_name, "value": metric_value} for metric_name, metric_value in metrics.items()]
            )

        return pd.DataFrame(rows)

    def _build_metadata(
        self,
        folds: list[dict[str, Any]],
        fold_metrics_rows: list[dict[str, Any]],
        strategy_summary: pd.DataFrame,
    ) -> dict[str, Any]:
        return {
            "num_folds": len(folds),
            "train_window": self.config.train_window,
            "validation_window": self.config.validation_window,
            "test_window": self.config.test_window,
            "step_size": self.config.step_size,
            "lookback_window": self.config.lookback_window,
            "commission_bps": self.config.commission_bps,
            "slippage_bps": self.config.slippage_bps,
            "phase5_max_epochs": self.config.phase5_max_epochs,
            "fold_metrics_preview": fold_metrics_rows[:4],
            "strategy_summary": strategy_summary.to_dict(orient="records"),
        }


class PhaseSixPipeline:
    """Persist Phase 6 rolling backtesting outputs and summaries."""

    def __init__(self, config: PhaseSixConfig | None = None) -> None:
        self.config = config or PhaseSixConfig()
        self.backtester = RollingWindowBacktester(self.config)

    def run(self, save_outputs: bool = True) -> PhaseSixArtifacts:
        artifacts = self.backtester.run()
        if save_outputs:
            self._write_outputs(artifacts)
        return artifacts

    def _write_outputs(self, artifacts: PhaseSixArtifacts) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        artifacts.fold_definitions.to_csv(self.config.output_dir / "fold_definitions.csv", index=False)
        artifacts.prediction_records.to_csv(self.config.output_dir / "prediction_records.csv", index=False)
        artifacts.fold_metrics.to_csv(self.config.output_dir / "fold_metrics.csv", index=False)
        artifacts.strategy_daily_returns.to_csv(self.config.output_dir / "strategy_daily_returns.csv", index=False)
        artifacts.strategy_summary.to_csv(self.config.output_dir / "strategy_summary.csv", index=False)
        with open(self.config.output_dir / "model_metadata.json", "w", encoding="utf-8") as handle:
            json.dump(artifacts.model_metadata, handle, indent=2)
