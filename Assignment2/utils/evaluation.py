"""
utils/evaluation.py
===================
Plotting and evaluation utilities shared across notebooks.
"""
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from pathlib import Path
 
 
PALETTE = {
    0: "#d62728",   # Strong Drop  → red
    1: "#ff7f0e",   # Mild Drop    → orange
    2: "#1f77b4",   # Neutral      → blue
    3: "#2ca02c",   # Mild Rise    → green
    4: "#17becf",   # Strong Rise  → teal
}
 
 
# ---------------------------------------------------------------------------
# Class distribution
# ---------------------------------------------------------------------------
 
def plot_class_distribution(
    y: pd.Series,
    class_names: dict,
    title: str = "Target Class Distribution",
    ax=None,
    save_path: Path | None = None,
):
    """Bar chart of class proportions with counts annotated."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()
 
    counts = y.value_counts().sort_index()
    pcts = counts / counts.sum() * 100
    labels = [class_names[i] for i in counts.index]
    colors = [PALETTE[i] for i in counts.index]
 
    bars = ax.bar(labels, pcts, color=colors, edgecolor="white", linewidth=0.8)
    for bar, cnt in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"n={cnt:,}",
            ha="center", va="bottom", fontsize=9
        )
    ax.set_ylabel("Proportion (%)")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax
 
 
# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
 
def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names: dict,
    title: str = "Confusion Matrix",
    ax=None,
    save_path: Path | None = None,
    normalize: str | None = "true",
):
    labels = sorted(class_names.keys())
    display_labels = [class_names[k] for k in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
 
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()
 
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format=".2f" if normalize else "d")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax
 
 
# ---------------------------------------------------------------------------
# Strategy / imputer comparison bar chart
# ---------------------------------------------------------------------------
 
def plot_f1_comparison(
    summary_df: pd.DataFrame,
    metric: str = "Macro F1",
    title: str = "Strategy Comparison",
    ax=None,
    save_path: Path | None = None,
    highlight_best: bool = True,
):
    """Horizontal bar chart comparing strategies by a chosen metric."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 0.55 * len(summary_df) + 1.5))
    else:
        fig = ax.get_figure()
 
    vals = summary_df[metric].sort_values()
    colors = ["#1f77b4"] * len(vals)
    if highlight_best:
        colors[-1] = "#d62728"
 
    ax.barh(vals.index, vals.values, color=colors, edgecolor="white")
    for i, (idx, val) in enumerate(vals.items()):
        ax.text(val + 0.002, i, f"{val:.4f}", va="center", fontsize=9)
    ax.set_xlabel(metric)
    ax.set_title(title)
    ax.set_xlim(0, vals.max() * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax
 
 
# ---------------------------------------------------------------------------
# Per-class F1 heatmap
# ---------------------------------------------------------------------------
 
def plot_per_class_f1_heatmap(
    summary_df: pd.DataFrame,
    class_names: dict,
    title: str = "Per-class F1 Score",
    save_path: Path | None = None,
):
    """Heatmap of F1 scores per class and per strategy."""
    f1_cols = [c for c in summary_df.columns if c.startswith("F1 [")]
    if not f1_cols:
        f1_cols = [c for c in summary_df.columns if "Class" in c or "F1" in c]
 
    data = summary_df[f1_cols].copy()
    fig, ax = plt.subplots(figsize=(len(f1_cols) * 1.4 + 1, len(data) * 0.55 + 1))
    sns.heatmap(
        data.astype(float),
        annot=True, fmt=".3f", cmap="RdYlGn",
        linewidths=0.5, linecolor="white",
        vmin=0, vmax=1, ax=ax
    )
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax
 
 
# ---------------------------------------------------------------------------
# Missing data heatmap
# ---------------------------------------------------------------------------
 
def plot_missing_heatmap(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    title: str = "Missing Values Map",
    save_path: Path | None = None,
    max_rows: int = 500,
):
    """Visualise where NaN values occur in the DataFrame."""
    if cols:
        data = df[cols]
    else:
        data = df.select_dtypes(include="number")
 
    # Sample if too large
    if len(data) > max_rows:
        data = data.sample(max_rows, random_state=42).sort_index()
 
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(
        data.isnull(),
        cbar=False,
        cmap=["white", "#d62728"],
        yticklabels=False,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Feature")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax