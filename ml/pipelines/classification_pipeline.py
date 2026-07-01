"""Classification ``Pipeline``: shared ``ColumnTransformer`` + estimator.

The default estimator (``LogisticRegression``) is a baseline only, proving
the preprocessing-to-estimator plumbing works end to end. Estimator
comparison and hyperparameter tuning happen in Milestones 9-10; this module
only wires the pieces together so that is possible.
"""

from __future__ import annotations

from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml.config import RANDOM_STATE
from ml.preprocessing.column_transformer import build_classification_transformer


def build_classification_pipeline(estimator: ClassifierMixin | None = None) -> Pipeline:
    """Assemble the classification ``Pipeline``: preprocessing + estimator.

    Args:
        estimator: A scikit-learn-compatible classifier. Defaults to
            ``LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)``
            if not provided.

    Returns:
        An unfitted ``Pipeline`` with steps ``"preprocess"`` and ``"model"``.
    """
    if estimator is None:
        estimator = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)

    return Pipeline(
        steps=[
            ("preprocess", build_classification_transformer()),
            ("model", estimator),
        ]
    )


if __name__ == "__main__":
    import logging

    from ml.data.feature_selection import select_classification_data
    from ml.data.split import get_train_test_split

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    train_df, test_df = get_train_test_split()
    X_train, y_train = select_classification_data(train_df)
    X_test, y_test = select_classification_data(test_df)

    pipeline = build_classification_pipeline()
    pipeline.fit(X_train, y_train)

    train_accuracy = pipeline.score(X_train, y_train)
    test_accuracy = pipeline.score(X_test, y_test)
    print(f"Baseline LogisticRegression -- train accuracy: {train_accuracy:.3f}, "
          f"test accuracy: {test_accuracy:.3f}")
