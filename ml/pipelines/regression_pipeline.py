"""Regression ``Pipeline``: shared ``ColumnTransformer`` + estimator.

Mirrors ``ml.pipelines.classification_pipeline``. The default estimator
(``Ridge``) is a baseline only, proving the preprocessing-to-estimator
plumbing works end to end. Estimator comparison and hyperparameter tuning
happen in Milestones 9-10.
"""

from __future__ import annotations

from sklearn.base import RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from ml.config import RANDOM_STATE
from ml.preprocessing.column_transformer import build_regression_transformer


def build_regression_pipeline(estimator: RegressorMixin | None = None) -> Pipeline:
    """Assemble the regression ``Pipeline``: preprocessing + estimator.

    Args:
        estimator: A scikit-learn-compatible regressor. Defaults to
            ``Ridge(random_state=RANDOM_STATE)`` if not provided.

    Returns:
        An unfitted ``Pipeline`` with steps ``"preprocess"`` and ``"model"``.
    """
    if estimator is None:
        estimator = Ridge(random_state=RANDOM_STATE)

    return Pipeline(
        steps=[
            ("preprocess", build_regression_transformer()),
            ("model", estimator),
        ]
    )


if __name__ == "__main__":
    import logging

    from sklearn.metrics import mean_absolute_error, r2_score

    from ml.data.feature_selection import select_regression_data
    from ml.data.split import get_train_test_split

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    train_df, test_df = get_train_test_split()
    X_train, y_train = select_regression_data(train_df)
    X_test, y_test = select_regression_data(test_df)

    pipeline = build_regression_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"Baseline Ridge -- test MAE: {mae:.3f} minutes, test R^2: {r2:.3f}")
