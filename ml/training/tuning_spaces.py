"""Candidate estimators and hyperparameter search spaces for both tasks.

Each entry pairs an unfitted estimator with a ``param_distributions`` dict
suitable for ``RandomizedSearchCV`` (keys prefixed with ``"model__"`` to
target the estimator step of the ``Pipeline`` built in Milestones 7-8).
Every estimator has ``random_state`` fixed for reproducibility.
"""

from __future__ import annotations

from typing import Any

from scipy.stats import loguniform, randint, uniform
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from ml.config import RANDOM_STATE

CLASSIFICATION_MODELS: dict[str, dict[str, Any]] = {
    # Naive baseline: every leaderboard must show how much the tuned models
    # actually beat "predict from the label distribution alone."
    "dummy": {
        "estimator": DummyClassifier(random_state=RANDOM_STATE),
        "param_distributions": {
            "model__strategy": ["most_frequent", "stratified", "prior"],
        },
    },
    "logistic_regression": {
        "estimator": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        "param_distributions": {
            "model__C": loguniform(1e-3, 1e2),
        },
    },
    "decision_tree": {
        "estimator": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "param_distributions": {
            "model__max_depth": randint(2, 20),
            "model__min_samples_leaf": randint(1, 20),
            "model__min_samples_split": randint(2, 20),
        },
    },
    "random_forest": {
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
        "param_distributions": {
            "model__n_estimators": randint(100, 400),
            "model__max_depth": randint(2, 20),
            "model__min_samples_leaf": randint(1, 20),
            "model__max_features": uniform(0.3, 0.7),
        },
    },
    "xgboost": {
        "estimator": XGBClassifier(
            random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=1
        ),
        "param_distributions": {
            "model__n_estimators": randint(100, 400),
            "model__max_depth": randint(2, 10),
            "model__learning_rate": loguniform(1e-2, 3e-1),
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.6, 0.4),
        },
    },
}

REGRESSION_MODELS: dict[str, dict[str, Any]] = {
    "dummy": {
        "estimator": DummyRegressor(),
        "param_distributions": {
            "model__strategy": ["mean", "median"],
        },
    },
    "ridge": {
        "estimator": Ridge(random_state=RANDOM_STATE),
        "param_distributions": {
            "model__alpha": loguniform(1e-2, 1e3),
        },
    },
    "decision_tree": {
        "estimator": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "param_distributions": {
            "model__max_depth": randint(2, 20),
            "model__min_samples_leaf": randint(1, 20),
            "model__min_samples_split": randint(2, 20),
        },
    },
    "random_forest": {
        "estimator": RandomForestRegressor(random_state=RANDOM_STATE),
        "param_distributions": {
            "model__n_estimators": randint(100, 400),
            "model__max_depth": randint(2, 20),
            "model__min_samples_leaf": randint(1, 20),
            "model__max_features": uniform(0.3, 0.7),
        },
    },
    "xgboost": {
        "estimator": XGBRegressor(random_state=RANDOM_STATE, n_jobs=1),
        "param_distributions": {
            "model__n_estimators": randint(100, 400),
            "model__max_depth": randint(2, 10),
            "model__learning_rate": loguniform(1e-2, 3e-1),
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.6, 0.4),
        },
    },
}
