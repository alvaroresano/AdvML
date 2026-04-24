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

            Folder scope for this notebook:

            - It only reads files from `Assignment1/financial_regression.csv`
            - `Assignment1/outputs/phase1/`
            - `Assignment1/outputs/phase2/`
            - `Assignment1/outputs/phase3/`

            The only file outside `Assignment1/` that relates to this workflow is `Project_Memoire.md`,
            but this notebook does not depend on it.
            """
        ),
        markdown_cell(
            """
            ## How To Read This Notebook

            The notebook is structured in five blocks:

            1. Data health and transformation overview
            2. Return behavior and stationarity
            3. Technical-indicator dashboards
            4. STL decomposition dashboards
            5. Classical SARIMAX baseline diagnostics

            If you need a quick oral explanation for class, the core story is:

            - prices trend strongly,
            - log returns are much more stationary,
            - technical indicators summarize momentum, trend, and local volatility,
            - STL shows that most assets have strong trend structure but weak weekly seasonality,
            - and the Phase 3 baseline checks whether a classical linear time-series model can explain the mean process well.
            """
        ),
        code_cell(
            """
            import json
            from pathlib import Path

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
            """
        ),
        code_cell(
            """
            phase3_metrics = phase3_meta["evaluation_metrics"]
            phase3_indicators = [
                ("ARIMA order", "".join(map(str, phase3_meta["order"]))),
                ("Seasonal order", ",".join(map(str, phase3_meta["seasonal_order"]))),
                ("Test RMSE", round(phase3_metrics["rmse"], 6)),
                ("Test MAE", round(phase3_metrics["mae"], 6)),
                ("Hit rate", round(phase3_metrics["directional_accuracy"], 4)),
                ("Test start", phase3_meta["test_start"]),
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
    ]
    return notebook


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    nbf.write(notebook, NOTEBOOK_PATH)
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
