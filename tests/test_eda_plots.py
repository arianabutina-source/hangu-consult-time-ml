"""Smoke tests for ml.eda.plots: each function must produce a non-empty PNG file."""

from pathlib import Path

import pandas as pd
import pytest

from ml.eda.plots import (
    generate_all_eda_figures,
    plot_categorical_counts,
    plot_missingness,
    plot_service_time_by_dayofweek,
    plot_service_time_distribution,
    plot_working_day_heatmap,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "DayOfWeek": ["Saturday", "Wednesday", "Friday", "Saturday", "Wednesday"] * 4,
            "WorkingDay": [False, True, True, False, True] * 4,
            "Gender": ["F", "M", "F", "M", "F"] * 4,
            "Address": ["In the city", "Out of city", None, "In the city", None] * 4,
            "ServTime": [600, 700, 800, 900, 1000] * 4,
        }
    )


def _assert_nonempty_png(path: Path) -> None:
    assert path.exists()
    assert path.suffix == ".png"
    assert path.stat().st_size > 0


def test_plot_missingness(sample_df: pd.DataFrame, tmp_path: Path) -> None:
    out = plot_missingness(sample_df, tmp_path / "missingness.png")
    _assert_nonempty_png(out)


def test_plot_service_time_distribution(sample_df: pd.DataFrame, tmp_path: Path) -> None:
    out = plot_service_time_distribution(sample_df, output_path=tmp_path / "dist.png")
    _assert_nonempty_png(out)


def test_plot_categorical_counts(sample_df: pd.DataFrame, tmp_path: Path) -> None:
    out = plot_categorical_counts(sample_df, "Gender", tmp_path / "gender.png")
    _assert_nonempty_png(out)


def test_plot_service_time_by_dayofweek(sample_df: pd.DataFrame, tmp_path: Path) -> None:
    out = plot_service_time_by_dayofweek(sample_df, output_path=tmp_path / "servtime_dow.png")
    _assert_nonempty_png(out)


def test_plot_working_day_heatmap(sample_df: pd.DataFrame, tmp_path: Path) -> None:
    out = plot_working_day_heatmap(sample_df, output_path=tmp_path / "heatmap.png")
    _assert_nonempty_png(out)


def test_generate_all_eda_figures(sample_df: pd.DataFrame, tmp_path: Path) -> None:
    paths = generate_all_eda_figures(sample_df, output_dir=tmp_path)
    assert len(paths) == 7
    for path in paths:
        _assert_nonempty_png(path)
