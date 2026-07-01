"""Shared ``ColumnTransformer`` builders for the classification and regression tasks.

These functions return **unfitted** transformers. Leakage prevention relies
on the caller fitting the returned transformer only on training data (e.g.
inside a ``Pipeline`` fit on ``X_train``) and only ever calling
``.transform`` -- never ``.fit``/``.fit_transform`` -- on the test set.

Three feature groups are handled differently:
    * numeric (``Visit.No``) -- median imputation + standard scaling.
    * boolean (already 0/1-valued flags) -- most-frequent imputation only;
      binary indicators are not standard-scaled, consistent with common
      practice for engineered/one-hot-like binary columns.
    * categorical -- most-frequent imputation + one-hot encoding, with
      unknown categories at inference time mapped to an all-zero row
      (``handle_unknown="ignore"``) rather than raising.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from ml.data.feature_selection import BOOLEAN_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES


def _cast_to_float(frame):
    """Cast boolean columns to float (module-level, not a lambda, so the
    fitted pipeline can be pickled -- see Milestone 13 serialization)."""
    return frame.astype(float)


def _build_column_transformer(
    numeric_features: list[str],
    boolean_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Assemble the shared numeric/boolean/categorical ``ColumnTransformer``."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    boolean_pipeline = Pipeline(
        steps=[
            # SimpleImputer does not accept boolean dtype directly.
            # feature_names_out="one-to-one" keeps get_feature_names_out()
            # working end to end (needed for Milestone 12 explainability).
            (
                "to_float",
                FunctionTransformer(_cast_to_float, feature_names_out="one-to-one"),
            ),
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("boolean", boolean_pipeline, boolean_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def build_classification_transformer() -> ColumnTransformer:
    """``ColumnTransformer`` for the classification task's predictor set."""
    return _build_column_transformer(NUMERIC_FEATURES, BOOLEAN_FEATURES, CATEGORICAL_FEATURES)


def build_regression_transformer() -> ColumnTransformer:
    """``ColumnTransformer`` for the regression task's predictor set.

    Currently identical to :func:`build_classification_transformer` because
    both tasks share the same predictor set (see
    ``ml.data.feature_selection``); kept as a separate entry point so the
    two tasks can diverge independently if either's feature set changes.
    """
    return _build_column_transformer(NUMERIC_FEATURES, BOOLEAN_FEATURES, CATEGORICAL_FEATURES)
