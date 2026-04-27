from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = REPO_ROOT / "Assignment1" / "02_Visual_Analytics.ipynb"


def markdown_cell(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip() + "\n")


def code_cell(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip() + "\n")


def build_notebook() -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook()
    notebook.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
        }
    )

    notebook.cells = [
        markdown_cell(
            """
            # Assignment 1 Visual Analytics Notebook

            This notebook is the visual companion to the work completed so far in `Assignment1/`.
            It focuses on:

            - the cleaned trading-calendar dataset,
            - return and technical-indicator features from Phase 1,
            - stationarity findings from the ADF tests,
            - STL decomposition outputs from Phase 2,
            - and the classical SARIMAX benchmark from Phase 3.
            - GARCH volatility outputs from Phase 4,
            - the PatchTST-style deep benchmark from Phase 5,
            - and the rolling backtesting results from Phase 6.

            Folder scope for this notebook:

            - It only reads files from `Assignment1/financial_regression.csv`
            - `Assignment1/outputs/phase1/`
            - `Assignment1/outputs/phase2/`
            - `Assignment1/outputs/phase3/`
            - `Assignment1/outputs/phase4/`
            - `Assignment1/outputs/phase5/`
            - `Assignment1/outputs/phase6/`

            The only file outside `Assignment1/` that relates to this workflow is `Project_Memoire.md`,
            but this notebook does not depend on it.
            """
        ),
        markdown_cell(
            """
            ## How To Read This Notebook

            The notebook is structured in eight blocks:

            1. Data health and transformation overview
            2. Return behavior and stationarity
            3. Technical-indicator dashboards
            4. STL decomposition dashboards
            5. Classical SARIMAX baseline diagnostics
            6. GARCH volatility diagnostics
            7. Deep-learning PatchTST-style benchmark
            8. Rolling backtesting and market-friction evaluation

            If you need a quick oral explanation for class, the core story is:

            - prices trend strongly,
            - log returns are much more stationary,
            - technical indicators summarize momentum, trend, and local volatility,
            - STL shows that most assets have strong trend structure but weak weekly seasonality,
            - the Phase 3 baseline checks whether a classical linear time-series model can explain the mean process well,
            - Phase 4 checks whether the remaining residual risk is conditionally heteroskedastic,
            - Phase 5 tests whether a modern sequence model can extract nonlinear structure beyond the classical benchmark,
            - and Phase 6 checks whether forecasting gains survive repeated historical evaluation once trading frictions are included.
            """
        ),
        markdown_cell(
            """
            ## Model Summary Table

            | Phase | Model / Procedure | Exact Specification Used | Main Role |
            |---|---|---|---|
            | Phase 3 | Classical benchmark | `SARIMAX` on `nasdaq log_return` with order `(0,0,0)`, seasonal order `(0,0,0,0)`, constant term, and 8 lagged exogenous regressors | Mean forecasting baseline |
            | Phase 4 | Volatility benchmark | `GARCH(1,1)` with Student-t innovations on Phase 3 residuals | Conditional variance / risk forecasting |
            | Phase 5 | Deep benchmark | PatchTST-style transformer with 60-day lookback, 10-day patches, 5-day stride, and 33 lagged features | Nonlinear sequence forecasting benchmark |
            | Phase 6 | Financial evaluation | 5-fold rolling walk-forward backtest with retraining, sign-based trading, 2 bps commissions, and 3 bps slippage | Economic usefulness under realistic frictions |

            This table is the shortest correct way to explain the modeling stack before going into details.
            """
        ),
        code_cell(
            """
            import json
            import os
            from pathlib import Path

            os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

            import numpy as np
            import pandas as pd
            import plotly.express as px
            import plotly.graph_objects as go
            import plotly.io as pio
            from plotly.subplots import make_subplots


            pio.templates.default = "plotly_white"
            px.defaults.template = "plotly_white"

            COLORWAY = [
                "#0F172A",
                "#2563EB",
                "#F97316",
                "#10B981",
                "#DC2626",
                "#7C3AED",
                "#D97706",
                "#0EA5E9",
            ]

            pio.templates["advml_theme"] = go.layout.Template(
                layout=go.Layout(
                    colorway=COLORWAY,
                    font=dict(family="Arial, sans-serif", size=13, color="#0F172A"),
                    title=dict(font=dict(size=22, color="#0F172A")),
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    legend=dict(
                        bgcolor="rgba(255,255,255,0.85)",
                        bordercolor="rgba(15,23,42,0.08)",
                        borderwidth=1,
                    ),
                    margin=dict(l=60, r=40, t=70, b=50),
                )
            )
            pio.templates.default = "advml_theme"


            def resolve_assignment1_dir() -> Path:
                cwd = Path.cwd().resolve()
                candidates = [
                    cwd,
                    cwd / "Assignment1",
                    cwd.parent / "Assignment1",
                ]
                for candidate in candidates:
                    if (candidate / "financial_regression.csv").exists():
                        return candidate
                raise FileNotFoundError(
                    "Could not locate Assignment1/. Open this notebook from the repo root or from inside Assignment1/."
                )


            ASSIGNMENT1_DIR = resolve_assignment1_dir()
            RAW_DATA_PATH = ASSIGNMENT1_DIR / "financial_regression.csv"
            PHASE1_DIR = ASSIGNMENT1_DIR / "outputs" / "phase1"
            PHASE2_DIR = ASSIGNMENT1_DIR / "outputs" / "phase2"
            PHASE3_DIR = ASSIGNMENT1_DIR / "outputs" / "phase3"
            PHASE4_DIR = ASSIGNMENT1_DIR / "outputs" / "phase4"
            PHASE5_DIR = ASSIGNMENT1_DIR / "outputs" / "phase5"
            PHASE6_DIR = ASSIGNMENT1_DIR / "outputs" / "phase6"

            raw_df = pd.read_csv(RAW_DATA_PATH, parse_dates=["date"]).sort_values("date")
            cleaned_df = pd.read_csv(PHASE1_DIR / "cleaned_data.csv", parse_dates=["date"]).sort_values("date")
            featured_df = pd.read_csv(PHASE1_DIR / "featured_data.csv", parse_dates=["date"]).sort_values("date")
            modeling_df = pd.read_csv(PHASE1_DIR / "modeling_data.csv", parse_dates=["date"]).sort_values("date")
            adf_df = pd.read_csv(PHASE1_DIR / "adf_summary.csv")
            stl_summary_df = pd.read_csv(PHASE2_DIR / "stl_decomposition_summary.csv")
            stl_components_df = pd.read_csv(PHASE2_DIR / "stl_decomposition_components.csv", parse_dates=["date"])
            phase3_design_df = pd.read_csv(PHASE3_DIR / "phase3_design_data.csv", parse_dates=["date"]).sort_values("date")
            phase3_train_df = pd.read_csv(PHASE3_DIR / "train_fitted.csv", parse_dates=["date"]).sort_values("date")
            phase3_test_df = pd.read_csv(PHASE3_DIR / "test_forecasts.csv", parse_dates=["date"]).sort_values("date")
            phase3_coeff_df = pd.read_csv(PHASE3_DIR / "coefficient_summary.csv")
            phase3_diag_df = pd.read_csv(PHASE3_DIR / "residual_diagnostics.csv")
            phase3_qq_df = pd.read_csv(PHASE3_DIR / "qq_plot_data.csv")
            with open(PHASE3_DIR / "model_metadata.json", "r", encoding="utf-8") as handle:
                phase3_meta = json.load(handle)
            phase4_train_df = pd.read_csv(PHASE4_DIR / "train_volatility.csv", parse_dates=["date"]).sort_values("date")
            phase4_test_df = pd.read_csv(PHASE4_DIR / "test_volatility_forecasts.csv", parse_dates=["date"]).sort_values("date")
            phase4_param_df = pd.read_csv(PHASE4_DIR / "garch_parameter_summary.csv")
            phase4_diag_df = pd.read_csv(PHASE4_DIR / "garch_residual_diagnostics.csv")
            phase4_qq_df = pd.read_csv(PHASE4_DIR / "garch_qq_plot_data.csv")
            with open(PHASE4_DIR / "garch_model_metadata.json", "r", encoding="utf-8") as handle:
                phase4_meta = json.load(handle)
            phase5_history_df = pd.read_csv(PHASE5_DIR / "training_history.csv")
            phase5_validation_df = pd.read_csv(PHASE5_DIR / "validation_predictions.csv", parse_dates=["date"]).sort_values("date")
            phase5_test_df = pd.read_csv(PHASE5_DIR / "test_predictions.csv", parse_dates=["date"]).sort_values("date")
            phase5_feature_schema_df = pd.read_csv(PHASE5_DIR / "feature_schema.csv")
            with open(PHASE5_DIR / "model_metadata.json", "r", encoding="utf-8") as handle:
                phase5_meta = json.load(handle)
            phase6_folds_df = pd.read_csv(PHASE6_DIR / "fold_definitions.csv")
            phase6_fold_metrics_df = pd.read_csv(PHASE6_DIR / "fold_metrics.csv")
            phase6_predictions_df = pd.read_csv(PHASE6_DIR / "prediction_records.csv", parse_dates=["date"]).sort_values(["model", "date"])
            phase6_strategy_df = pd.read_csv(PHASE6_DIR / "strategy_daily_returns.csv", parse_dates=["date"]).sort_values(["model", "date"])
            phase6_summary_df = pd.read_csv(PHASE6_DIR / "strategy_summary.csv")
            with open(PHASE6_DIR / "model_metadata.json", "r", encoding="utf-8") as handle:
                phase6_meta = json.load(handle)

            assets = [column.removesuffix(" close") for column in cleaned_df.columns if column.endswith(" close")]
            close_columns = [f"{asset} close" for asset in assets]
            log_return_columns = [f"{asset} log_return" for asset in assets]

            print(f"Assignment1 directory: {ASSIGNMENT1_DIR}")
            print(f"Assets: {assets}")
            print(f"Raw shape: {raw_df.shape}")
            print(f"Cleaned shape: {cleaned_df.shape}")
            print(f"Featured shape: {featured_df.shape}")
            print(f"Modeling shape: {modeling_df.shape}")
            print(f"Phase 3 train/test: {phase3_meta['train_rows']} / {phase3_meta['test_rows']}")
            print(f"Phase 4 persistence: {phase4_meta['persistence']:.4f}")
            print(f"Phase 5 test RMSE: {phase5_meta['evaluation_metrics']['rmse']:.6f}")
            print(f"Phase 6 folds: {phase6_meta['num_folds']}")
            """
        ),
        markdown_cell(
            """
            ## 1. Data Health And Transformation Overview

            Before interpreting any model or indicator, it is important to know what happened to the raw data.

            The visuals below answer four practical questions:

            - How many rows were removed and why?
            - How much of the raw panel was driven by non-trading dates?
            - How do the asset price levels compare once normalized?
            - Which transformed variables were created for later modeling?
            """
        ),
        code_cell(
            """
            rows_removed_before_start = int((raw_df["date"] < pd.Timestamp("2010-04-01")).sum())
            non_trading_rows_removed = int(
                raw_df.loc[raw_df["date"] >= pd.Timestamp("2010-04-01"), close_columns].isna().all(axis=1).sum()
            )

            metrics = [
                ("Raw Rows", len(raw_df)),
                ("Rows After Clean", len(cleaned_df)),
                ("Modeling Rows", len(modeling_df)),
                ("Assets", len(assets)),
                ("Rows Trimmed Pre-2010-04-01", rows_removed_before_start),
                ("Non-Trading Rows Removed", non_trading_rows_removed),
            ]

            fig = make_subplots(
                rows=1,
                cols=len(metrics),
                specs=[[{"type": "indicator"} for _ in metrics]],
            )

            for index, (label, value) in enumerate(metrics, start=1):
                fig.add_trace(
                    go.Indicator(
                        mode="number",
                        value=value,
                        title={"text": label},
                        number={"font": {"size": 34}},
                    ),
                    row=1,
                    col=index,
                )

            fig.update_layout(
                title="Pipeline Snapshot After Phase 1 And Phase 2",
                height=240,
            )
            fig.show()
            """
        ),
        code_cell(
            """
            non_trading_dates = raw_df.loc[
                (raw_df["date"] >= pd.Timestamp("2010-04-01")) & raw_df[close_columns].isna().all(axis=1),
                ["date"],
            ].assign(event="non-trading row")

            row_flow = pd.DataFrame(
                {
                    "stage": ["Raw dataset", "After start-date trim", "After non-trading filter", "Complete-case modeling"],
                    "rows": [len(raw_df), len(raw_df) - rows_removed_before_start, len(cleaned_df), len(modeling_df)],
                }
            )

            fig = make_subplots(
                rows=1,
                cols=2,
                column_widths=[0.45, 0.55],
                subplot_titles=("Row Count Through The Pipeline", "Where The Non-Trading Gaps Occur"),
            )

            fig.add_trace(
                go.Bar(
                    x=row_flow["stage"],
                    y=row_flow["rows"],
                    marker_color=["#0F172A", "#2563EB", "#10B981", "#F97316"],
                    text=row_flow["rows"],
                    textposition="outside",
                    name="rows",
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=non_trading_dates["date"],
                    y=np.ones(len(non_trading_dates)),
                    mode="markers",
                    marker=dict(size=8, color="#DC2626", opacity=0.75),
                    name="removed non-trading dates",
                    hovertemplate="%{x|%Y-%m-%d}<extra></extra>",
                ),
                row=1,
                col=2,
            )

            fig.update_yaxes(title_text="Rows", row=1, col=1)
            fig.update_yaxes(visible=False, row=1, col=2)
            fig.update_xaxes(title_text="", row=1, col=1)
            fig.update_xaxes(title_text="Date", row=1, col=2)
            fig.update_layout(height=450, title="Data Reduction And Trading-Calendar Cleaning")
            fig.show()
            """
        ),
        code_cell(
            """
            normalized_prices = cleaned_df.set_index("date")[close_columns].div(
                cleaned_df.set_index("date")[close_columns].iloc[0]
            ) * 100

            normalized_long = (
                normalized_prices.reset_index()
                .melt(id_vars="date", var_name="asset", value_name="normalized_index")
                .assign(asset=lambda frame: frame["asset"].str.replace(" close", "", regex=False))
            )

            fig = px.line(
                normalized_long,
                x="date",
                y="normalized_index",
                color="asset",
                title="Normalized Close Prices (Base 100 On First Cleaned Trading Day)",
                labels={"normalized_index": "Index level", "date": "Date", "asset": "Asset"},
            )
            fig.update_layout(height=550)
            fig.show()
            """
        ),
        markdown_cell(
            """
            ## 2. Log Returns, Correlation, And Stationarity

            A price series is often hard to model directly because it drifts over time.
            Log returns are more useful for inference because they are closer to a stationary process.

            In simple terms:

            - prices answer: "what is the market level?"
            - log returns answer: "how much did the market move from one day to the next?"
            """
        ),
        code_cell(
            """
            returns_long = (
                featured_df[["date", *log_return_columns]]
                .melt(id_vars="date", var_name="asset", value_name="log_return")
                .dropna()
                .assign(asset=lambda frame: frame["asset"].str.replace(" log_return", "", regex=False))
            )

            fig = px.line(
                returns_long,
                x="date",
                y="log_return",
                color="asset",
                facet_row="asset",
                facet_row_spacing=0.005,
                height=1400,
                title="Log Returns Through Time",
                labels={"log_return": "Log return", "date": "Date", "asset": ""},
            )
            fig.for_each_annotation(lambda annotation: annotation.update(text=annotation.text.split("=")[-1]))
            fig.update_layout(showlegend=False)
            fig.show()
            """
        ),
        code_cell(
            """
            fig = px.histogram(
                returns_long,
                x="log_return",
                color="asset",
                facet_col="asset",
                facet_col_wrap=2,
                nbins=70,
                opacity=0.7,
                title="Distribution Of Log Returns By Asset",
                labels={"log_return": "Log return", "asset": "Asset"},
                height=1200,
            )
            fig.for_each_annotation(lambda annotation: annotation.update(text=annotation.text.split("=")[-1]))
            fig.update_layout(showlegend=False)
            fig.show()
            """
        ),
        code_cell(
            """
            correlation_columns = log_return_columns + ["GDP", "CPI", "us_rates_%", "usd_chf", "eur_usd"]
            correlation_frame = featured_df[correlation_columns].dropna().rename(
                columns=lambda column: column.replace(" log_return", "")
            )
            corr = correlation_frame.corr()

            fig = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                aspect="auto",
                title="Correlation Heatmap: Asset Log Returns, Macro Variables, And FX",
            )
            fig.update_layout(height=850)
            fig.show()
            """
        ),
        code_cell(
            """
            adf_plot = adf_df.copy()
            adf_plot["display_p_value"] = adf_plot["p_value"].clip(lower=1e-20)
            adf_plot["minus_log10_p"] = -np.log10(adf_plot["display_p_value"])
            adf_plot["transformation_label"] = adf_plot["transformation"].map(
                {"price_level": "Price level", "log_return": "Log return"}
            )

            fig = px.bar(
                adf_plot,
                x="asset",
                y="minus_log10_p",
                color="transformation_label",
                barmode="group",
                title="ADF Evidence Against A Unit Root",
                labels={
                    "asset": "Asset",
                    "minus_log10_p": "-log10(p-value)",
                    "transformation_label": "Series type",
                },
            )
            fig.add_hline(
                y=-np.log10(0.05),
                line_dash="dash",
                line_color="#DC2626",
                annotation_text="5% significance threshold",
            )
            fig.update_layout(height=520)
            fig.show()

            adf_display = adf_df[["asset", "transformation", "adf_statistic", "p_value", "reject_unit_root_5pct"]].copy()
            adf_display
            """
        ),
        markdown_cell(
            """
            ### How To Explain The ADF Chart

            A taller bar means stronger evidence against the unit-root null.

            In your results:

            - the price-level bars stay below the significance threshold,
            - while the log-return bars are far above it.

            That is exactly the pattern expected in finance:

            - price levels behave like non-stationary processes,
            - log returns are much closer to stationary processes.
            """
        ),
        markdown_cell(
            """
            ## 3. Technical-Indicator Dashboards

            These charts help translate raw price action into interpretable signals:

            - **Bollinger Bands**: local price level relative to rolling volatility
            - **RSI(14)**: momentum pressure on a 0-100 scale
            - **MACD(12,26,9)**: short-term momentum against longer-term momentum

            A simple explanation:

            - If price presses the upper band, it is high relative to recent volatility.
            - If RSI moves above 70, upside momentum may be stretched.
            - If the MACD line crosses above the signal line, short-term momentum is strengthening.
            """
        ),
        code_cell(
            """
            def build_indicator_dashboard(asset: str) -> go.Figure:
                asset_frame = featured_df[[
                    "date",
                    f"{asset} open",
                    f"{asset} high",
                    f"{asset} low",
                    f"{asset} close",
                    f"{asset} log_return",
                    f"{asset} rsi_14",
                    f"{asset} macd_line",
                    f"{asset} macd_signal",
                    f"{asset} macd_hist",
                    f"{asset} bb_upper",
                    f"{asset} bb_middle",
                    f"{asset} bb_lower",
                ]].copy()

                fig = make_subplots(
                    rows=4,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=[0.45, 0.15, 0.18, 0.22],
                    subplot_titles=(
                        f"{asset.upper()} price with Bollinger Bands",
                        "Daily log returns",
                        "RSI(14)",
                        "MACD(12,26,9)",
                    ),
                )

                fig.add_trace(
                    go.Candlestick(
                        x=asset_frame["date"],
                        open=asset_frame[f"{asset} open"],
                        high=asset_frame[f"{asset} high"],
                        low=asset_frame[f"{asset} low"],
                        close=asset_frame[f"{asset} close"],
                        name="OHLC",
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=asset_frame["date"],
                        y=asset_frame[f"{asset} bb_upper"],
                        mode="lines",
                        line=dict(width=1.2, color="#DC2626"),
                        name="Bollinger upper",
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=asset_frame["date"],
                        y=asset_frame[f"{asset} bb_middle"],
                        mode="lines",
                        line=dict(width=1.2, color="#2563EB", dash="dot"),
                        name="Bollinger middle",
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=asset_frame["date"],
                        y=asset_frame[f"{asset} bb_lower"],
                        mode="lines",
                        line=dict(width=1.2, color="#10B981"),
                        name="Bollinger lower",
                        fill="tonexty",
                        fillcolor="rgba(37, 99, 235, 0.08)",
                    ),
                    row=1,
                    col=1,
                )

                log_return = asset_frame[f"{asset} log_return"]
                fig.add_trace(
                    go.Bar(
                        x=asset_frame["date"],
                        y=log_return,
                        marker_color=np.where(log_return >= 0, "#10B981", "#DC2626"),
                        name="Log return",
                    ),
                    row=2,
                    col=1,
                )

                fig.add_trace(
                    go.Scatter(
                        x=asset_frame["date"],
                        y=asset_frame[f"{asset} rsi_14"],
                        mode="lines",
                        line=dict(color="#7C3AED", width=1.6),
                        name="RSI(14)",
                    ),
                    row=3,
                    col=1,
                )
                fig.add_hline(y=70, line_dash="dash", line_color="#DC2626", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="#10B981", row=3, col=1)

                macd_hist = asset_frame[f"{asset} macd_hist"]
                fig.add_trace(
                    go.Bar(
                        x=asset_frame["date"],
                        y=macd_hist,
                        marker_color=np.where(macd_hist >= 0, "#2563EB", "#F97316"),
                        name="MACD histogram",
                    ),
                    row=4,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=asset_frame["date"],
                        y=asset_frame[f"{asset} macd_line"],
                        mode="lines",
                        line=dict(color="#0F172A", width=1.6),
                        name="MACD line",
                    ),
                    row=4,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=asset_frame["date"],
                        y=asset_frame[f"{asset} macd_signal"],
                        mode="lines",
                        line=dict(color="#DC2626", width=1.4, dash="dot"),
                        name="MACD signal",
                    ),
                    row=4,
                    col=1,
                )
                fig.add_hline(y=0, line_dash="dot", line_color="#64748B", row=4, col=1)

                fig.update_yaxes(title_text="Price", row=1, col=1)
                fig.update_yaxes(title_text="Return", row=2, col=1)
                fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
                fig.update_yaxes(title_text="MACD", row=4, col=1)
                fig.update_layout(height=1100, title=f"{asset.upper()} technical dashboard", xaxis_rangeslider_visible=False)
                return fig
            """
        ),
        code_cell(
            """
            for asset in assets:
                build_indicator_dashboard(asset).show()
            """
        ),
        markdown_cell(
            """
            ## 4. STL Decomposition

            STL separates each log-price series into:

            - observed log price,
            - smooth trend,
            - seasonal component,
            - residual shock component.

            The point of this section is not only to make pretty charts.
            It is to answer a modeling question:

            **Are these series mainly trend-driven, seasonal, or shock-driven?**
            """
        ),
        code_cell(
            """
            summary_long = stl_summary_df.melt(
                id_vars="asset",
                value_vars=["trend_strength", "seasonal_strength", "residual_share_of_variance"],
                var_name="metric",
                value_name="value",
            )

            metric_labels = {
                "trend_strength": "Trend strength",
                "seasonal_strength": "Seasonal strength",
                "residual_share_of_variance": "Residual share of variance",
            }
            summary_long["metric"] = summary_long["metric"].map(metric_labels)

            fig = px.bar(
                summary_long,
                x="asset",
                y="value",
                color="metric",
                barmode="group",
                title="STL Summary Metrics By Asset",
                labels={"asset": "Asset", "value": "Score", "metric": "Metric"},
            )
            fig.update_layout(height=560)
            fig.show()
            """
        ),
        code_cell(
            """
            fig = px.bar(
                stl_summary_df.sort_values("seasonal_amplitude", ascending=False),
                x="asset",
                y="seasonal_amplitude",
                color="asset",
                title="Seasonal Amplitude From STL",
                labels={"asset": "Asset", "seasonal_amplitude": "Amplitude of seasonal component"},
            )
            fig.update_layout(height=500, showlegend=False)
            fig.show()

            stl_summary_df.sort_values("asset")
            """
        ),
        code_cell(
            """
            def build_stl_dashboard(asset: str) -> go.Figure:
                asset_frame = stl_components_df.loc[stl_components_df["asset"] == asset].copy()

                fig = make_subplots(
                    rows=4,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=[0.34, 0.22, 0.18, 0.18],
                    subplot_titles=(
                        f"{asset.upper()} observed log price",
                        "Trend",
                        "Seasonal component",
                        "Residual component",
                    ),
                )

                fig.add_trace(
                    go.Scatter(
                        x=asset_frame["date"],
                        y=asset_frame["observed_log_price"],
                        mode="lines",
                        line=dict(color="#0F172A", width=1.4),
                        name="Observed log price",
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=asset_frame["date"],
                        y=asset_frame["trend"],
                        mode="lines",
                        line=dict(color="#2563EB", width=1.8),
                        name="Trend",
                    ),
                    row=2,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=asset_frame["date"],
                        y=asset_frame["seasonal"],
                        mode="lines",
                        line=dict(color="#F97316", width=1.4),
                        name="Seasonal",
                    ),
                    row=3,
                    col=1,
                )
                fig.add_hline(y=0, line_dash="dot", line_color="#64748B", row=3, col=1)
                fig.add_trace(
                    go.Scatter(
                        x=asset_frame["date"],
                        y=asset_frame["residual"],
                        mode="lines",
                        line=dict(color="#10B981", width=1.2),
                        name="Residual",
                    ),
                    row=4,
                    col=1,
                )
                fig.add_hline(y=0, line_dash="dot", line_color="#64748B", row=4, col=1)

                fig.update_yaxes(title_text="log(P)", row=1, col=1)
                fig.update_yaxes(title_text="Trend", row=2, col=1)
                fig.update_yaxes(title_text="Seasonal", row=3, col=1)
                fig.update_yaxes(title_text="Residual", row=4, col=1)
                fig.update_layout(height=980, title=f"{asset.upper()} STL decomposition")
                return fig
            """
        ),
        code_cell(
            """
            for asset in assets:
                build_stl_dashboard(asset).show()
            """
        ),
        markdown_cell(
            """
            ## What The STL Visuals Are Saying

            The summary bars and detailed dashboards support the same conclusion:

            - The **trend** component dominates most assets.
            - The **seasonal** component is weak for nearly all assets under a 5-trading-day cycle.
            - The **residual** component captures shocks, abrupt moves, and market-specific irregular behavior.

            A good plain-language interpretation is:

            "These markets do not look strongly driven by a stable weekly seasonal pattern.
            They look mostly driven by long-term trend plus unpredictable shocks."

            That is why the next modeling phases should focus on:

            - autoregressive structure,
            - exogenous macro effects,
            - and volatility clustering,

            rather than assuming strong deterministic seasonality from the start.
            """
        ),
        markdown_cell(
            """
            ## 5. Classical SARIMAX Baseline

            Phase 3 moves from exploration to a formal benchmark model.

            The key question is:

            **After converting prices to returns and adding lagged exogenous information, is there still meaningful linear time-series structure left in the mean process?**

            For the current benchmark:

            - the target is `nasdaq log_return`,
            - the exogenous block uses lagged market, FX, and macro-change variables,
            - `auto_arima` chooses the order with AIC,
            - and the selected structure is then refit as a SARIMAX model for diagnostics and forecasting.

            The explicit answer to "Are we using ARIMA or SARIMAX?" is:

            - we are using **SARIMAX** as the actual benchmark model,
            - while `auto_arima` is only the **order-selection tool**.

            Since exogenous regressors are included, the correct model name is `SARIMAX`, not plain `ARIMA`.
            """
        ),
        code_cell(
            """
            phase3_metrics = phase3_meta["evaluation_metrics"]
            phase3_indicators = [
                ("Train rows", phase3_meta["train_rows"]),
                ("Test rows", phase3_meta["test_rows"]),
                ("Test RMSE", round(phase3_metrics["rmse"], 6)),
                ("Test MAE", round(phase3_metrics["mae"], 6)),
                ("Hit rate", round(phase3_metrics["directional_accuracy"], 4)),
                ("Mean error", round(phase3_metrics["mean_forecast_error"], 6)),
            ]

            fig = make_subplots(
                rows=1,
                cols=len(phase3_indicators),
                specs=[[{"type": "indicator"} for _ in phase3_indicators]],
            )

            for index, (label, value) in enumerate(phase3_indicators, start=1):
                fig.add_trace(
                    go.Indicator(
                        mode="number",
                        value=value if isinstance(value, (int, float)) else 0,
                        number={"font": {"size": 32}, "valueformat": ".4f"},
                        title={"text": f"{label}<br><span style='font-size:15px'>{value}</span>"},
                    ),
                    row=1,
                    col=index,
                )

            fig.update_layout(height=240, title="Phase 3 Benchmark Snapshot")
            fig.show()

            pd.DataFrame(
                {
                    "item": ["Model class", "ARIMA order", "Seasonal order", "Trend", "Number of exogenous variables", "Test window"],
                    "value": [
                        "SARIMAX",
                        tuple(phase3_meta["order"]),
                        tuple(phase3_meta["seasonal_order"]),
                        phase3_meta["trend"],
                        len(phase3_meta["exogenous_columns"]),
                        f"{phase3_meta['test_start']} to {phase3_meta['test_end']}",
                    ],
                }
            )
            """
        ),
        code_cell(
            """
            split_date = phase3_test_df["date"].min()

            combined_fit = pd.concat(
                [
                    phase3_train_df.assign(segment="train"),
                    phase3_test_df.rename(columns={"forecast": "fitted"}).assign(segment="test"),
                ],
                ignore_index=True,
            )

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=phase3_train_df["date"],
                    y=phase3_train_df["actual"],
                    mode="lines",
                    line=dict(color="#0F172A", width=1.3),
                    name="Train actual",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=phase3_train_df["date"],
                    y=phase3_train_df["fitted"],
                    mode="lines",
                    line=dict(color="#2563EB", width=1.1),
                    name="Train fitted",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=phase3_test_df["date"],
                    y=phase3_test_df["actual"],
                    mode="lines",
                    line=dict(color="#10B981", width=1.4),
                    name="Test actual",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=phase3_test_df["date"],
                    y=phase3_test_df["forecast"],
                    mode="lines",
                    line=dict(color="#F97316", width=1.5),
                    name="Test forecast",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=phase3_test_df["date"],
                    y=phase3_test_df["upper_ci"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=phase3_test_df["date"],
                    y=phase3_test_df["lower_ci"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(249,115,22,0.16)",
                    name="95% interval",
                )
            )
            fig.add_shape(
                type="line",
                x0=split_date,
                x1=split_date,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                line=dict(color="#DC2626", dash="dash"),
            )
            fig.add_annotation(
                x=split_date,
                y=1.02,
                xref="x",
                yref="paper",
                text="Train / test split",
                showarrow=False,
                font=dict(color="#DC2626"),
            )
            fig.update_layout(
                title="NASDAQ Log Returns: In-Sample Fit And Out-Of-Sample Forecast",
                height=560,
                yaxis_title="Log return",
                xaxis_title="Date",
            )
            fig.show()
            """
        ),
        code_cell(
            """
            coeff_plot = phase3_coeff_df.query("parameter != 'sigma2'").copy()
            coeff_plot["significance"] = np.where(coeff_plot["p_value"] < 0.05, "p < 0.05", "not significant")

            fig = px.bar(
                coeff_plot.sort_values("coefficient"),
                x="coefficient",
                y="parameter",
                color="significance",
                orientation="h",
                title="Phase 3 Coefficients With Significance Highlighting",
                labels={"coefficient": "Estimated coefficient", "parameter": "Parameter"},
            )
            for _, row in coeff_plot.iterrows():
                fig.add_shape(
                    type="line",
                    x0=row["ci_lower"],
                    x1=row["ci_upper"],
                    y0=row["parameter"],
                    y1=row["parameter"],
                    line=dict(color="#0F172A", width=2),
                )
            fig.add_vline(x=0, line_dash="dot", line_color="#64748B")
            fig.update_layout(height=520)
            fig.show()

            phase3_coeff_df.sort_values("p_value")
            """
        ),
        code_cell(
            """
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "Training residuals through time",
                    "Residual distribution",
                    "Q-Q plot against normal reference",
                    "Ljung-Box p-values by lag",
                ),
            )

            fig.add_trace(
                go.Scatter(
                    x=phase3_train_df["date"],
                    y=phase3_train_df["residual"],
                    mode="lines",
                    line=dict(color="#2563EB", width=1.0),
                    name="Residual",
                ),
                row=1,
                col=1,
            )
            fig.add_hline(y=0, line_dash="dot", line_color="#64748B", row=1, col=1)

            fig.add_trace(
                go.Histogram(
                    x=phase3_train_df["residual"],
                    nbinsx=70,
                    marker_color="#F97316",
                    name="Residual histogram",
                ),
                row=1,
                col=2,
            )

            fig.add_trace(
                go.Scatter(
                    x=phase3_qq_df["theoretical_quantile"],
                    y=phase3_qq_df["sample_quantile"],
                    mode="markers",
                    marker=dict(color="#0EA5E9", size=5, opacity=0.65),
                    name="Q-Q points",
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=phase3_qq_df["theoretical_quantile"],
                    y=phase3_qq_df["reference_line"],
                    mode="lines",
                    line=dict(color="#DC2626", width=1.5),
                    name="Reference line",
                ),
                row=2,
                col=1,
            )

            ljung_plot = phase3_diag_df[phase3_diag_df["lag"] != "jarque_bera"].copy()
            ljung_plot["lag"] = ljung_plot["lag"].astype(int)
            fig.add_trace(
                go.Bar(
                    x=ljung_plot["lag"],
                    y=ljung_plot["ljung_box_pvalue"],
                    marker_color="#10B981",
                    name="Ljung-Box p-value",
                ),
                row=2,
                col=2,
            )
            fig.add_hline(y=0.05, line_dash="dash", line_color="#DC2626", row=2, col=2)

            fig.update_xaxes(title_text="Date", row=1, col=1)
            fig.update_xaxes(title_text="Residual", row=1, col=2)
            fig.update_xaxes(title_text="Theoretical quantile", row=2, col=1)
            fig.update_xaxes(title_text="Lag", row=2, col=2)
            fig.update_yaxes(title_text="Residual", row=1, col=1)
            fig.update_yaxes(title_text="Count", row=1, col=2)
            fig.update_yaxes(title_text="Sample quantile", row=2, col=1)
            fig.update_yaxes(title_text="p-value", row=2, col=2)
            fig.update_layout(height=900, title="Residual Diagnostics For The Phase 3 Benchmark")
            fig.show()

            phase3_diag_df
            """
        ),
        markdown_cell(
            """
            ## What The Phase 3 Visuals Are Saying

            The classical benchmark produces three important lessons:

            1. `auto_arima` selects an order of `(0,0,0)` for the NASDAQ return target once lagged exogenous variables are included.
            2. This means the mean process does not show strong additional ARMA structure after the return transformation and exogenous block are introduced.
            3. However, the residual diagnostics are still not fully clean:
               - Ljung-Box p-values become very small at longer lags,
               - the Q-Q plot departs from the straight line in the tails,
               - and Jarque-Bera rejects normality strongly.

            A clear classroom explanation is:

            "The classical mean model captures part of the signal, but the residuals still contain dependence and heavy tails.
            So the next stages should focus on modeling conditional variance and richer forecasting structure, not just the conditional mean."
            """
        ),
        markdown_cell(
            """
            ## 6. GARCH Volatility Diagnostics

            Phase 4 asks a different question from Phase 3.

            Phase 3 modeled the **conditional mean** of returns.
            Phase 4 models the **conditional variance** of the remaining errors.

            In plain language:

            - the mean model asks: "What is the expected return?"
            - the GARCH model asks: "How much risk or volatility should we expect around that mean?"

            This matters in finance because volatility is not constant. Quiet periods and turbulent periods tend to cluster.
            """
        ),
        code_cell(
            """
            phase4_metrics = phase4_meta["evaluation_metrics"]
            phase4_indicators = [
                ("Persistence", round(phase4_meta["persistence"], 4)),
                ("Uncond. vol", round(phase4_meta["unconditional_volatility"], 4)),
                ("Half-life", round(phase4_meta["half_life_periods"], 2)),
                ("nu", round(phase4_meta["nu"], 4)),
                ("QLIKE", round(phase4_metrics["qlike"], 4)),
                ("Vol RMSE", round(phase4_metrics["volatility_rmse"], 4)),
            ]

            fig = make_subplots(
                rows=1,
                cols=len(phase4_indicators),
                specs=[[{"type": "indicator"} for _ in phase4_indicators]],
            )

            for index, (label, value) in enumerate(phase4_indicators, start=1):
                fig.add_trace(
                    go.Indicator(
                        mode="number",
                        value=value,
                        title={"text": label},
                        number={"font": {"size": 30}},
                    ),
                    row=1,
                    col=index,
                )

            fig.update_layout(height=240, title="Phase 4 GARCH Risk Snapshot")
            fig.show()
            """
        ),
        code_cell(
            """
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.06,
                subplot_titles=(
                    "Phase 3 mean residuals",
                    "Phase 4 conditional volatility estimated by GARCH(1,1)",
                ),
            )

            fig.add_trace(
                go.Scatter(
                    x=phase4_train_df["date"],
                    y=phase4_train_df["mean_residual"],
                    mode="lines",
                    line=dict(color="#0F172A", width=1.0),
                    name="Mean residual",
                ),
                row=1,
                col=1,
            )
            fig.add_hline(y=0, line_dash="dot", line_color="#64748B", row=1, col=1)

            fig.add_trace(
                go.Scatter(
                    x=phase4_train_df["date"],
                    y=phase4_train_df["conditional_volatility"],
                    mode="lines",
                    line=dict(color="#DC2626", width=1.5),
                    name="Conditional volatility",
                ),
                row=2,
                col=1,
            )

            fig.update_yaxes(title_text="Residual", row=1, col=1)
            fig.update_yaxes(title_text="Volatility", row=2, col=1)
            fig.update_layout(height=800, title="Why We Need GARCH: Shocks Cluster In Time")
            fig.show()
            """
        ),
        code_cell(
            """
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=phase4_test_df["date"],
                    y=phase4_test_df["forecast_volatility"],
                    mode="lines",
                    line=dict(color="#DC2626", width=1.6),
                    name="Forecast volatility",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=phase4_test_df["date"],
                    y=np.sqrt(phase4_test_df["realized_sq_error"]),
                    mode="lines",
                    line=dict(color="#2563EB", width=1.0),
                    name="Realized absolute error proxy",
                )
            )
            fig.update_layout(
                title="Out-of-Sample Volatility Forecast vs Realized Absolute Error Proxy",
                height=540,
                yaxis_title="Volatility scale",
                xaxis_title="Date",
            )
            fig.show()
            """
        ),
        code_cell(
            """
            fig = px.bar(
                phase4_param_df,
                x="parameter",
                y="estimate",
                color="parameter",
                title="Estimated GARCH(1,1)-t Parameters",
                labels={"estimate": "Estimate", "parameter": "Parameter"},
            )
            for _, row in phase4_param_df.iterrows():
                fig.add_shape(
                    type="line",
                    x0=row["parameter"],
                    x1=row["parameter"],
                    y0=row["ci_lower"],
                    y1=row["ci_upper"],
                    line=dict(color="#0F172A", width=2),
                )
            fig.update_layout(height=500, showlegend=False)
            fig.show()

            phase4_param_df
            """
        ),
        code_cell(
            """
            ljung_phase4 = phase4_diag_df[phase4_diag_df["test"].str.contains("ljung_box")].copy()
            ljung_phase4["series_type"] = np.where(
                ljung_phase4["test"].str.contains("squared"),
                "Squared standardized residual",
                "Standardized residual",
            )

            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "Standardized residuals through time",
                    "Standardized residual distribution",
                    "Q-Q plot against fitted Student-t reference",
                    "Ljung-Box p-values after GARCH filtering",
                ),
            )

            fig.add_trace(
                go.Scatter(
                    x=phase4_train_df["date"],
                    y=phase4_train_df["standardized_residual"],
                    mode="lines",
                    line=dict(color="#2563EB", width=1.0),
                    name="Standardized residual",
                ),
                row=1,
                col=1,
            )
            fig.add_hline(y=0, line_dash="dot", line_color="#64748B", row=1, col=1)

            fig.add_trace(
                go.Histogram(
                    x=phase4_train_df["standardized_residual"],
                    nbinsx=60,
                    marker_color="#F97316",
                    name="Standardized residual histogram",
                ),
                row=1,
                col=2,
            )

            fig.add_trace(
                go.Scatter(
                    x=phase4_qq_df["theoretical_quantile_t"],
                    y=phase4_qq_df["sample_quantile"],
                    mode="markers",
                    marker=dict(size=5, color="#0EA5E9", opacity=0.6),
                    name="Q-Q points",
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=phase4_qq_df["theoretical_quantile_t"],
                    y=phase4_qq_df["reference_line"],
                    mode="lines",
                    line=dict(color="#DC2626", width=1.5),
                    name="Reference line",
                ),
                row=2,
                col=1,
            )

            for series_type, color in [
                ("Standardized residual", "#2563EB"),
                ("Squared standardized residual", "#10B981"),
            ]:
                subset = ljung_phase4[ljung_phase4["series_type"] == series_type]
                fig.add_trace(
                    go.Bar(
                        x=subset["lag"],
                        y=subset["p_value"],
                        name=series_type,
                        marker_color=color,
                    ),
                    row=2,
                    col=2,
                )
            fig.add_hline(y=0.05, line_dash="dash", line_color="#DC2626", row=2, col=2)

            fig.update_xaxes(title_text="Date", row=1, col=1)
            fig.update_xaxes(title_text="Standardized residual", row=1, col=2)
            fig.update_xaxes(title_text="Theoretical t quantile", row=2, col=1)
            fig.update_xaxes(title_text="Lag", row=2, col=2)
            fig.update_yaxes(title_text="Std. residual", row=1, col=1)
            fig.update_yaxes(title_text="Count", row=1, col=2)
            fig.update_yaxes(title_text="Sample quantile", row=2, col=1)
            fig.update_yaxes(title_text="p-value", row=2, col=2)
            fig.update_layout(height=920, title="Diagnostic Check After GARCH Filtering")
            fig.show()

            phase4_diag_df
            """
        ),
        markdown_cell(
            """
            ## What The Phase 4 Visuals Are Saying

            The most important GARCH lesson is this:

            - the model does **not** try to predict the sign of the residual,
            - it tries to predict the **scale of uncertainty** around the mean model.

            In the current results:

            - persistence is very high, around `0.9916`,
            - which means volatility shocks decay slowly,
            - the half-life is roughly `82` trading periods,
            - and the Student-t degrees of freedom parameter is around `5.89`, which is consistent with heavy tails.

            The diagnostic improvement is also specific:

            - the squared standardized residual tests improve strongly,
            - which means the volatility clustering is being captured much better,
            - but the standardized residuals themselves still show some serial structure,
            - and the residual distribution is still not perfectly Gaussian.

            A clean classroom explanation is:

            "The GARCH model materially improves the variance dynamics. It removes most of the autocorrelation from squared residuals, which means it is capturing volatility clustering well. But it does not solve every remaining modeling issue, so the process is still not fully iid after filtering."
            """
        ),
        markdown_cell(
            """
            ## 7. Deep Learning Benchmark: PatchTST-Style Forecasting

            Phase 5 returns to the **mean forecasting** problem, but now with a nonlinear sequence model.

            The idea behind a PatchTST-style architecture is:

            - instead of feeding one day at a time into the transformer, we group the history into short temporal patches,
            - each patch becomes a token,
            - self-attention then learns which historical fragments are most informative for the next-step forecast.

            In this project, the deep model uses:

            - a `60`-day lookback window,
            - patch length `10`,
            - patch stride `5`,
            - `33` lagged multivariate features,
            - and a one-step-ahead NASDAQ log-return target.

            This is a rigorous local benchmark because it can be trained reproducibly inside the repository, unlike a much larger external foundation model that would require downloading pretrained weights.
            """
        ),
        markdown_cell(
            """
            ### Phase 5 Glossary

            These are the most important terms to explain clearly in class:

            - **Lookback window**: how many past trading days the model sees before making one forecast. Here it is `60`.
            - **Patch length**: how many consecutive days are grouped into one token. Here it is `10`.
            - **Patch stride**: how far the patch window moves each time. Here it is `5`, so patches overlap.
            - **Token**: the basic unit processed by the transformer. In this notebook, one token is one 10-day time patch.
            - **Channel**: one feature observed through time, such as NASDAQ return, gold RSI, or oil Bollinger z-score.
            - **RMSE**: error metric that penalizes large misses more strongly.
            - **MAE**: average absolute forecast miss.
            - **Hit Rate**: fraction of days for which the forecast sign matches the realized return sign.
            - **Forecast/actual correlation**: linear alignment between predictions and realized outcomes.

            For a non-finance audience, a good one-sentence summary is:

            "The deep model looks at the last 60 trading days, compresses that history into overlapping 10-day chunks, and uses attention to decide which recent patterns matter most for tomorrow's NASDAQ return."
            """
        ),
        markdown_cell(
            """
            ### What The Code Is Doing

            The implemented model is split into two layers:

            - `PatchTSTForecaster`: the neural network itself
            - `PatchTSTDeepForecaster`: the data-preparation, training, prediction, and evaluation pipeline

            The pipeline logic is:

            1. load the Phase 1 modeling table,
            2. build a lagged multivariate design matrix,
            3. standardize inputs using training data only,
            4. turn the table into rolling 60-day supervised windows,
            5. train the transformer with early stopping,
            6. reload the best validation checkpoint,
            7. forecast the holdout period,
            8. compare the resulting errors against the classical Phase 3 baseline.

            The network logic is:

            1. reshape each input window into feature-by-time format,
            2. create overlapping patches with `unfold`,
            3. embed each patch into a learned latent vector,
            4. add positional information,
            5. run the patch sequence through the transformer encoder,
            6. pool the encoded representation,
            7. map that representation to one scalar next-day forecast.
            """
        ),
        markdown_cell(
            """
            ### Where This Architecture Comes From

            This model was **not invented from scratch**. It is a compact adaptation of the ideas from the PatchTST paper,
            *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*, and its official repository.

            The original PatchTST design highlights two main ideas:

            - **patching** the time axis so each token is a short subseries rather than one raw point,
            - **channel-independence** so channels share embedding and transformer weights.

            Our implementation keeps those core ideas:

            - overlapping temporal patches,
            - shared patch embedding across channels,
            - shared transformer encoder across channels,
            - and final aggregation into one supervised forecast.

            But it is also a deliberate simplification:

            - it is a CPU-friendly course-project adaptation,
            - it is built for one-step-ahead NASDAQ return forecasting,
            - and it uses our financial feature set rather than reproducing the full official benchmark stack.

            So the rigorous description is:

            "This is a PatchTST-style architectural adaptation inspired by the paper and official implementation, not a line-by-line reproduction of the authors' original experiment code."
            """
        ),
        code_cell(
            """
            phase5_metrics = phase5_meta["evaluation_metrics"]
            phase5_benchmark = phase5_meta.get("phase3_benchmark", {})
            phase5_indicators = [
                ("PatchTST RMSE", round(phase5_metrics["rmse"], 5)),
                ("PatchTST MAE", round(phase5_metrics["mae"], 5)),
                ("PatchTST Hit Rate", round(phase5_metrics["directional_accuracy"], 4)),
                ("Correlation", round(phase5_metrics["forecast_actual_correlation"], 4)),
                ("Best Epoch", int(phase5_meta["best_epoch"])),
                ("Features", int(phase5_meta["feature_count"])),
            ]

            fig = make_subplots(
                rows=1,
                cols=len(phase5_indicators),
                specs=[[{"type": "indicator"} for _ in phase5_indicators]],
            )

            for index, (label, value) in enumerate(phase5_indicators, start=1):
                fig.add_trace(
                    go.Indicator(
                        mode="number",
                        value=value,
                        title={"text": label},
                        number={"font": {"size": 30}},
                    ),
                    row=1,
                    col=index,
                )

            fig.update_layout(height=240, title="Phase 5 Deep Forecasting Snapshot")
            fig.show()

            comparison_df = pd.DataFrame(
                {
                    "metric": ["RMSE", "MAE", "Hit Rate"],
                    "Phase 3 SARIMAX": [
                        phase5_benchmark.get("rmse", np.nan),
                        phase5_benchmark.get("mae", np.nan),
                        phase5_benchmark.get("directional_accuracy", np.nan),
                    ],
                    "Phase 5 PatchTST": [
                        phase5_metrics["rmse"],
                        phase5_metrics["mae"],
                        phase5_metrics["directional_accuracy"],
                    ],
                }
            )
            comparison_df
            """
        ),
        code_cell(
            """
            comparison_long = comparison_df.melt(id_vars="metric", var_name="model", value_name="value")
            fig = px.bar(
                comparison_long,
                x="metric",
                y="value",
                color="model",
                barmode="group",
                title="Phase 3 vs Phase 5 Holdout Metrics",
                labels={"value": "Metric value", "metric": "Metric", "model": "Model"},
            )
            fig.update_layout(height=500)
            fig.show()
            """
        ),
        code_cell(
            """
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=phase5_history_df["epoch"],
                    y=phase5_history_df["train_rmse"],
                    mode="lines+markers",
                    line=dict(color="#2563EB", width=2),
                    name="Train RMSE",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=phase5_history_df["epoch"],
                    y=phase5_history_df["validation_rmse"],
                    mode="lines+markers",
                    line=dict(color="#DC2626", width=2),
                    name="Validation RMSE",
                )
            )
            fig.add_vline(
                x=phase5_meta["best_epoch"],
                line_dash="dash",
                line_color="#0F172A",
                annotation_text="best epoch",
            )
            fig.update_layout(
                title="Training Curve For The PatchTST-Style Model",
                height=500,
                xaxis_title="Epoch",
                yaxis_title="RMSE on scaled target",
            )
            fig.show()
            """
        ),
        code_cell(
            """
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                subplot_titles=(
                    "Out-of-sample NASDAQ return forecast vs realized return",
                    "Forecast errors through time",
                ),
            )

            fig.add_trace(
                go.Scatter(
                    x=phase5_test_df["date"],
                    y=phase5_test_df["actual"],
                    mode="lines",
                    line=dict(color="#0F172A", width=1.4),
                    name="Actual return",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=phase5_test_df["date"],
                    y=phase5_test_df["forecast"],
                    mode="lines",
                    line=dict(color="#2563EB", width=1.6),
                    name="PatchTST forecast",
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Bar(
                    x=phase5_test_df["date"],
                    y=phase5_test_df["forecast_error"],
                    marker_color=np.where(phase5_test_df["forecast_error"] >= 0, "#10B981", "#DC2626"),
                    name="Forecast error",
                ),
                row=2,
                col=1,
            )
            fig.add_hline(y=0, line_dash="dot", line_color="#64748B", row=2, col=1)

            fig.update_yaxes(title_text="Return", row=1, col=1)
            fig.update_yaxes(title_text="Error", row=2, col=1)
            fig.update_layout(height=820, title="Phase 5 Holdout Forecast Path")
            fig.show()
            """
        ),
        code_cell(
            """
            fig = make_subplots(
                rows=1,
                cols=3,
                subplot_titles=(
                    "Actual vs forecast scatter",
                    "Forecast distribution",
                    "Feature mix by category",
                ),
            )

            fig.add_trace(
                go.Scatter(
                    x=phase5_test_df["actual"],
                    y=phase5_test_df["forecast"],
                    mode="markers",
                    marker=dict(size=7, color="#2563EB", opacity=0.65),
                    name="Actual vs forecast",
                ),
                row=1,
                col=1,
            )
            line_min = min(phase5_test_df["actual"].min(), phase5_test_df["forecast"].min())
            line_max = max(phase5_test_df["actual"].max(), phase5_test_df["forecast"].max())
            fig.add_trace(
                go.Scatter(
                    x=[line_min, line_max],
                    y=[line_min, line_max],
                    mode="lines",
                    line=dict(color="#DC2626", dash="dash"),
                    name="45-degree line",
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Histogram(
                    x=phase5_test_df["forecast"],
                    nbinsx=40,
                    marker_color="#F97316",
                    name="Forecast distribution",
                ),
                row=1,
                col=2,
            )

            feature_mix = phase5_feature_schema_df["category"].value_counts().sort_values(ascending=False).reset_index()
            feature_mix.columns = ["category", "count"]
            fig.add_trace(
                go.Bar(
                    x=feature_mix["category"],
                    y=feature_mix["count"],
                    marker_color="#10B981",
                    name="Feature count",
                ),
                row=1,
                col=3,
            )

            fig.update_xaxes(title_text="Actual", row=1, col=1)
            fig.update_xaxes(title_text="Forecast", row=1, col=2)
            fig.update_xaxes(title_text="Category", row=1, col=3)
            fig.update_yaxes(title_text="Forecast", row=1, col=1)
            fig.update_yaxes(title_text="Count", row=1, col=2)
            fig.update_yaxes(title_text="Features", row=1, col=3)
            fig.update_layout(height=520, title="What The Deep Model Learned To Use And Predict")
            fig.show()

            phase5_feature_schema_df.head(12)
            """
        ),
        markdown_cell(
            """
            ## What The Phase 5 Visuals Are Saying

            The deep model improves the classical benchmark, but only modestly:

            - test RMSE improves from about `0.01103` in Phase 3 to about `0.01090`,
            - test MAE improves from about `0.00824` to about `0.00810`,
            - and hit rate improves from about `0.528` to about `0.571`.

            That is a real improvement, but it must be interpreted honestly.

            The model is still forecasting very small daily returns, and the forecast distribution remains compressed around zero.
            This is typical in high-noise financial return forecasting: even a useful model often explains only a small fraction of the realized next-day move.

            Two points matter most for interpretation:

            - the training curve shows that validation performance stops improving much earlier than training performance, so early stopping is important,
            - and the actual-vs-forecast scatter shows weak but nonzero alignment rather than a tight line, which means the model is extracting some structure but not producing large-confidence return calls.

            A clean classroom explanation is:

            "The deep model is better than the classical linear benchmark on the holdout sample, but the improvement is incremental rather than dramatic. That is realistic for daily financial return prediction, where the signal-to-noise ratio is low. The result supports using modern sequence models, but it also shows that no model here should be presented as if it had strong deterministic forecasting power."
            """
        ),
        markdown_cell(
            """
            ### Why Phase 6 Is Still Necessary

            It is true that by the end of Phase 5 we already have:

            - a classical benchmark,
            - a volatility model,
            - a deep-learning benchmark,
            - and a fixed holdout comparison.

            But that is **not** the same thing as a full financial evaluation.

            A single train/test split can be misleading because performance may depend heavily on one specific market regime.
            Phase 6 therefore asks a stricter question:

            "If we repeated the forecast-and-trade exercise many times through history using only information available at each date, would the strategy remain useful after costs and frictions?"

            That is the purpose of rolling-window backtesting. It turns a single benchmark comparison into a proper historical simulation.
            """
        ),
        markdown_cell(
            """
            ## 8. Rolling Backtesting And Market Frictions

            Phase 6 is where forecasting becomes a financial evaluation rather than only a statistical one.

            The rolling backtest used here does the following:

            1. define a training window, validation window, and test window,
            2. retrain the model on each fold,
            3. generate out-of-sample forecasts for the next historical block,
            4. convert forecasts into trading positions using the sign of the forecast,
            5. subtract slippage and commissions when the position changes,
            6. compute both forecast metrics and trading-performance metrics.

            This matters because a model can improve RMSE while still failing to improve realized trading performance.
            """
        ),
        code_cell(
            """
            def get_summary_metric(model: str, metric: str) -> float:
                value = phase6_summary_df.loc[
                    (phase6_summary_df["model"] == model) & (phase6_summary_df["metric"] == metric),
                    "value",
                ]
                return float(value.iloc[0])

            phase6_snapshot = pd.DataFrame(
                {
                    "metric": ["Net Cum. Return", "Sharpe", "Max Drawdown", "Hit Rate"],
                    "Phase 3 SARIMAX": [
                        get_summary_metric("phase3_sarimax", "net_cumulative_return"),
                        get_summary_metric("phase3_sarimax", "annualized_sharpe_ratio"),
                        get_summary_metric("phase3_sarimax", "maximum_drawdown"),
                        get_summary_metric("phase3_sarimax", "directional_accuracy"),
                    ],
                    "Phase 5 PatchTST": [
                        get_summary_metric("phase5_patchtst", "net_cumulative_return"),
                        get_summary_metric("phase5_patchtst", "annualized_sharpe_ratio"),
                        get_summary_metric("phase5_patchtst", "maximum_drawdown"),
                        get_summary_metric("phase5_patchtst", "directional_accuracy"),
                    ],
                }
            )
            phase6_snapshot
            """
        ),
        code_cell(
            """
            phase6_snapshot_long = phase6_snapshot.melt(id_vars="metric", var_name="model", value_name="value")
            fig = px.bar(
                phase6_snapshot_long,
                x="metric",
                y="value",
                color="model",
                barmode="group",
                title="Phase 6 Backtest Summary: Forecasting vs Trading KPIs",
                labels={"value": "Metric value", "metric": "Metric", "model": "Model"},
            )
            fig.update_layout(height=520)
            fig.show()
            """
        ),
        code_cell(
            """
            fig = go.Figure()
            for model_name, label, color in [
                ("phase3_sarimax", "Phase 3 SARIMAX", "#0F172A"),
                ("phase5_patchtst", "Phase 5 PatchTST", "#2563EB"),
            ]:
                subset = phase6_strategy_df[phase6_strategy_df["model"] == model_name]
                fig.add_trace(
                    go.Scatter(
                        x=subset["date"],
                        y=subset["cumulative_net"],
                        mode="lines",
                        line=dict(width=2, color=color),
                        name=label,
                    )
                )
            fig.update_layout(
                title="Cumulative Net Wealth After Slippage And Commissions",
                height=560,
                xaxis_title="Date",
                yaxis_title="Cumulative wealth index",
            )
            fig.show()
            """
        ),
        code_cell(
            """
            fold_plot = phase6_fold_metrics_df.copy()
            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=("Fold-by-fold RMSE", "Fold-by-fold directional accuracy"),
            )

            for model_name, label, color in [
                ("phase3_sarimax", "Phase 3 SARIMAX", "#0F172A"),
                ("phase5_patchtst", "Phase 5 PatchTST", "#2563EB"),
            ]:
                subset = fold_plot[fold_plot["model"] == model_name]
                fig.add_trace(
                    go.Scatter(
                        x=subset["fold_id"],
                        y=subset["rmse"],
                        mode="lines+markers",
                        line=dict(color=color, width=2),
                        name=label,
                        legendgroup=label,
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=subset["fold_id"],
                        y=subset["directional_accuracy"],
                        mode="lines+markers",
                        line=dict(color=color, width=2),
                        name=label,
                        legendgroup=label,
                        showlegend=False,
                    ),
                    row=1,
                    col=2,
                )

            fig.update_xaxes(title_text="Fold", row=1, col=1)
            fig.update_xaxes(title_text="Fold", row=1, col=2)
            fig.update_yaxes(title_text="RMSE", row=1, col=1)
            fig.update_yaxes(title_text="Directional accuracy", row=1, col=2)
            fig.update_layout(height=480, title="Rolling-Fold Prediction Stability")
            fig.show()
            """
        ),
        code_cell(
            """
            turnover_plot = phase6_strategy_df.groupby("model", as_index=False)["turnover"].mean()
            turnover_plot["label"] = turnover_plot["model"].map(
                {"phase3_sarimax": "Phase 3 SARIMAX", "phase5_patchtst": "Phase 5 PatchTST"}
            )

            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=("Average turnover", "Distribution of daily net returns"),
            )
            fig.add_trace(
                go.Bar(
                    x=turnover_plot["label"],
                    y=turnover_plot["turnover"],
                    marker_color=["#0F172A", "#2563EB"],
                    name="Average turnover",
                ),
                row=1,
                col=1,
            )
            for model_name, label, color in [
                ("phase3_sarimax", "Phase 3 SARIMAX", "#0F172A"),
                ("phase5_patchtst", "Phase 5 PatchTST", "#2563EB"),
            ]:
                subset = phase6_strategy_df[phase6_strategy_df["model"] == model_name]
                fig.add_trace(
                    go.Histogram(
                        x=subset["net_return"],
                        opacity=0.60,
                        nbinsx=70,
                        marker_color=color,
                        name=label,
                    ),
                    row=1,
                    col=2,
                )

            fig.update_xaxes(title_text="Model", row=1, col=1)
            fig.update_xaxes(title_text="Daily net return", row=1, col=2)
            fig.update_yaxes(title_text="Turnover", row=1, col=1)
            fig.update_yaxes(title_text="Count", row=1, col=2)
            fig.update_layout(height=520, barmode="overlay", title="Trading Friction And Return Distribution")
            fig.show()
            """
        ),
        markdown_cell(
            """
            ## What The Phase 6 Visuals Are Saying

            Phase 6 delivers the most important practical lesson of the whole project:

            - better forecast metrics do **not automatically** imply better trading performance.

            In the saved backtest:

            - PatchTST is slightly better on RMSE, MAE, and hit rate,
            - but SARIMAX ends with a slightly higher net cumulative return and Sharpe ratio,
            - while PatchTST suffers a slightly deeper maximum drawdown.

            That is a very realistic financial result.

            It means the deep model is statistically competitive, but its trading signal is still not economically dominant once repeated walk-forward evaluation and market frictions are introduced.

            A clean classroom explanation is:

            "Single-split forecasting results suggested that the deep model was stronger. Rolling backtesting shows a more nuanced picture: the deep model improves some prediction metrics, but those gains do not translate cleanly into superior historical trading performance after costs. This is why financial machine learning must evaluate both predictive accuracy and economic utility."
            """
        ),
    ]
    return notebook


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    nbf.write(notebook, NOTEBOOK_PATH)
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
