"""Tests for scripts.export_dashboard_data: JSON shape/content sanity checks."""

import json

import pytest

from ml.config import CLASSIFICATION_TUNING_RESULTS_PATH
from scripts.export_dashboard_data import DASHBOARD_DIR, main


@pytest.mark.skipif(
    not CLASSIFICATION_TUNING_RESULTS_PATH.exists(), reason="tuning results not yet generated"
)
def test_export_dashboard_data_produces_expected_files() -> None:
    main()

    expected_files = [
        "confusion_matrix.json",
        "roc_curve.json",
        "feature_importance_classification.json",
        "feature_importance_regression.json",
        "residuals.json",
    ]
    for filename in expected_files:
        path = DASHBOARD_DIR / filename
        assert path.exists(), f"missing {filename}"
        json.loads(path.read_text())  # must parse as valid JSON


@pytest.mark.skipif(
    not CLASSIFICATION_TUNING_RESULTS_PATH.exists(), reason="tuning results not yet generated"
)
def test_confusion_matrix_json_shape() -> None:
    main()
    data = json.loads((DASHBOARD_DIR / "confusion_matrix.json").read_text())
    assert data["labels"] == ["Short", "Long"]
    assert len(data["matrix"]) == 2
    assert all(len(row) == 2 for row in data["matrix"])


@pytest.mark.skipif(
    not CLASSIFICATION_TUNING_RESULTS_PATH.exists(), reason="tuning results not yet generated"
)
def test_roc_curve_json_is_monotonic_in_fpr() -> None:
    main()
    points = json.loads((DASHBOARD_DIR / "roc_curve.json").read_text())
    fprs = [p["fpr"] for p in points]
    assert fprs == sorted(fprs)
    assert all(0.0 <= p["fpr"] <= 1.0 and 0.0 <= p["tpr"] <= 1.0 for p in points)


@pytest.mark.skipif(
    not CLASSIFICATION_TUNING_RESULTS_PATH.exists(), reason="tuning results not yet generated"
)
def test_feature_importance_json_is_sorted_descending() -> None:
    main()
    records = json.loads((DASHBOARD_DIR / "feature_importance_classification.json").read_text())
    importances = [r["importance"] for r in records]
    assert importances == sorted(importances, reverse=True)
    assert len(records) <= 10
