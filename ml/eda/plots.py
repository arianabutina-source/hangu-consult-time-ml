"""Figure generation for exploratory data analysis.

Each function renders one figure, saves it as a PNG, closes the figure to
avoid accumulating open matplotlib state, and returns the output path.
Figures are saved under ``config.REPORT_FIGURES_DIR`` by default so the
Quarto report (Milestone 19) can embed them directly without recomputation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ml.config import RAW_SERVICE_TIME_COLUMN, REPORT_FIGURES_DIR, SECONDS_PER_MINUTE
from ml.eda.summary import missing_value_report, working_day_crosstab

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")


def _save_and_close(fig: plt.Figure, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure to %s", output_path)
    return output_path


def plot_missingness(
    df: pd.DataFrame, output_path: Path = REPORT_FIGURES_DIR / "eda_missingness.png"
) -> Path:
    """Bar chart of the percentage of missing values per column (columns with any missing only)."""
    report = missing_value_report(df)
    report = report[report["n_missing"] > 0]

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=report["pct_missing"], y=report.index, hue=report.index, legend=False, ax=ax)
    ax.set_xlabel("Missing (%)")
    ax.set_ylabel("")
    ax.set_title("Missing Values by Column")
    return _save_and_close(fig, output_path)


def plot_service_time_distribution(
    df: pd.DataFrame,
    seconds_column: str = RAW_SERVICE_TIME_COLUMN,
    output_path: Path = REPORT_FIGURES_DIR / "eda_servtime_distribution.png",
) -> Path:
    """Histogram (with KDE) of consultation duration, converted to minutes."""
    minutes = df[seconds_column] / SECONDS_PER_MINUTE

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(minutes, bins=40, kde=True, ax=ax)
    ax.set_xlabel("Consultation duration (minutes)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Consultation Duration")
    return _save_and_close(fig, output_path)


def plot_categorical_counts(
    df: pd.DataFrame, column: str, output_path: Path | None = None
) -> Path:
    """Bar chart of value counts for a categorical column."""
    if output_path is None:
        output_path = REPORT_FIGURES_DIR / f"eda_counts_{column.replace('.', '_').lower()}.png"

    counts = df[column].value_counts()

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=counts.values, y=counts.index, hue=counts.index, legend=False, ax=ax)
    ax.set_xlabel("Count")
    ax.set_ylabel(column)
    ax.set_title(f"Distribution of {column}")
    return _save_and_close(fig, output_path)


def plot_service_time_by_dayofweek(
    df: pd.DataFrame,
    seconds_column: str = RAW_SERVICE_TIME_COLUMN,
    day_column: str = "DayOfWeek",
    output_path: Path = REPORT_FIGURES_DIR / "eda_servtime_by_dayofweek.png",
) -> Path:
    """Boxplot of consultation duration (minutes) grouped by day of week."""
    plot_df = df[[day_column, seconds_column]].copy()
    plot_df["minutes"] = plot_df[seconds_column] / SECONDS_PER_MINUTE

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=plot_df, x=day_column, y="minutes", hue=day_column, legend=False, ax=ax)
    ax.set_xlabel("Day of week")
    ax.set_ylabel("Consultation duration (minutes)")
    ax.set_title("Consultation Duration by Day of Week")
    return _save_and_close(fig, output_path)


def plot_working_day_heatmap(
    df: pd.DataFrame,
    day_column: str = "DayOfWeek",
    working_day_column: str = "WorkingDay",
    output_path: Path = REPORT_FIGURES_DIR / "eda_workingday_heatmap.png",
) -> Path:
    """Heatmap of the DayOfWeek x WorkingDay contingency table."""
    crosstab = working_day_crosstab(df, day_column, working_day_column)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(crosstab, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Day of Week vs. Working Day")
    return _save_and_close(fig, output_path)


def generate_all_eda_figures(
    df: pd.DataFrame, output_dir: Path = REPORT_FIGURES_DIR
) -> list[Path]:
    """Generate and save the full standard set of EDA figures.

    Args:
        df: Raw (or cleaned) DataFrame to plot.
        output_dir: Directory figures are written to.

    Returns:
        List of paths to the saved figures.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_missingness(df, output_dir / "eda_missingness.png"),
        plot_service_time_distribution(df, output_path=output_dir / "eda_servtime_distribution.png"),
        plot_categorical_counts(df, "DayOfWeek", output_dir / "eda_counts_dayofweek.png"),
        plot_categorical_counts(df, "Gender", output_dir / "eda_counts_gender.png"),
        plot_categorical_counts(df, "Address", output_dir / "eda_counts_address.png"),
        plot_service_time_by_dayofweek(df, output_path=output_dir / "eda_servtime_by_dayofweek.png"),
        plot_working_day_heatmap(df, output_path=output_dir / "eda_workingday_heatmap.png"),
    ]
    logger.info("Generated %d EDA figures in %s", len(paths), output_dir)
    return paths
