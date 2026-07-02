"""Project-wide constants: paths, reproducibility settings, and the raw dataset schema.

Centralising these values means every module validates against the same
expectations instead of re-declaring "magic" numbers, and reproducibility
(``RANDOM_STATE``) is set in exactly one place.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
RAW_DATA_PATH: Path = RAW_DATA_DIR / "Data.csv"
REPORT_DIR: Path = PROJECT_ROOT / "report"
REPORT_FIGURES_DIR: Path = REPORT_DIR / "figures"

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_STATE: int = 42

# --------------------------------------------------------------------------- #
# Raw dataset schema (established during dataset inspection)
# --------------------------------------------------------------------------- #
EXPECTED_N_ROWS: int = 6_637
EXPECTED_N_COLUMNS: int = 14

EXPECTED_COLUMNS: list[str] = [
    "ID",
    "Session",
    "Month",
    "DayOfWeek",
    "WorkingDay",
    "AM_PM",
    "Visit.No",
    "Gender",
    "M.Cancer",
    "S.Cancer",
    "StartTime",
    "PayTime",
    "Address",
    "ServTime",
]

# Columns whose pandas dtype is checked after loading. Columns not listed
# here (e.g. free-form strings) are intentionally left unvalidated.
EXPECTED_NUMERIC_COLUMNS: list[str] = ["Session", "Visit.No", "ServTime"]
EXPECTED_BOOLEAN_COLUMNS: list[str] = ["WorkingDay", "M.Cancer", "S.Cancer"]

# Consultation duration, as recorded in the raw data (seconds).
RAW_SERVICE_TIME_COLUMN: str = "ServTime"
SECONDS_PER_MINUTE: int = 60

# Patient identifier used to group repeat visits (must never be used as a
# model predictor, only for grouped splitting/aggregation).
PATIENT_ID_COLUMN: str = "ID"

# --------------------------------------------------------------------------- #
# Data cleaning (Milestone 3)
# --------------------------------------------------------------------------- #
# Address is missing for ~36% of rows (see EDA). Rather than impute or drop,
# missingness is treated as its own informative category, since "unknown
# address" is plausibly meaningful for an outpatient scheduling context.
ADDRESS_COLUMN: str = "Address"
ADDRESS_MISSING_LABEL: str = "Unknown"

# Expected value domains for categorical columns, used to guard against
# silent data drift (e.g. a new category appearing in a future data refresh).
EXPECTED_GENDER_VALUES: set[str] = {"F", "M"}
EXPECTED_MONTH_VALUES: set[str] = {
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
}
EXPECTED_DAYOFWEEK_VALUES: set[str] = {"Friday", "Saturday", "Tuesday", "Wednesday"}
EXPECTED_AMPM_VALUES: set[str] = {"morning", "afternoon"}
EXPECTED_ADDRESS_VALUES: set[str] = {
    "In the city", "Out of city", "Out of province", ADDRESS_MISSING_LABEL,
}

# PayTime is missing for ~4% of rows. EDA showed PayTime - StartTime does not
# reliably reconstruct ServTime, so these timestamp columns are not used to
# derive or validate the target and are left untouched (missing values kept
# as NaN) rather than imputed.
TIMESTAMP_COLUMNS: list[str] = ["StartTime", "PayTime"]

# --------------------------------------------------------------------------- #
# Feature engineering & targets (Milestone 4)
# --------------------------------------------------------------------------- #
# Regression target: ServTime converted from seconds to minutes.
SERVICE_TIME_MINUTES_COLUMN: str = "ServTime_minutes"
REGRESSION_TARGET_COLUMN: str = SERVICE_TIME_MINUTES_COLUMN

# Classification target: consultations longer than a fixed cutoff.
#
# The cutoff is the dataset-wide MEDIAN duration, computed once from the
# full historical population and then frozen as a constant business-rule
# threshold (documented below) -- analogous to a fixed clinical cutoff
# (e.g. "BMI > 30 is obese"). It is not refit per train/test split or per
# CV fold, so it is a label-definition choice, not a leaking preprocessing
# statistic. Using the median also yields close to a balanced 50/50 class
# split, which keeps classification metrics (accuracy, F1, ROC-AUC)
# meaningful. The exact computed value is logged whenever features are
# engineered and must be reported in the Quarto report (Milestone 19).
CLASSIFICATION_TARGET_COLUMN: str = "IsLongConsultation"
CLASSIFICATION_THRESHOLD_METHOD: str = "median"

# Additional engineered features (row-level, no cross-row/patient
# aggregation, so no leakage risk):
#   - IsRepeatVisit: Visit.No > 1
#   - IsPublicHolidayMakeup: WorkingDay is True on a day that is normally
#     non-working for this clinic (Saturday) -- operationalises the
#     "public holiday" predictor named in the original project
#     requirements, using the compensatory-Saturday pattern found in EDA.
#   - HasCancerDiagnosis: M.Cancer OR S.Cancer, since S.Cancer alone is
#     rare (~1% True) and sparse as a standalone predictor.
IS_REPEAT_VISIT_COLUMN: str = "IsRepeatVisit"
VISIT_NO_COLUMN: str = "Visit.No"

IS_PUBLIC_HOLIDAY_MAKEUP_COLUMN: str = "IsPublicHolidayMakeup"
NORMALLY_NON_WORKING_DAY: str = "Saturday"

HAS_CANCER_DIAGNOSIS_COLUMN: str = "HasCancerDiagnosis"

# --------------------------------------------------------------------------- #
# Train/test split (Milestone 6)
# --------------------------------------------------------------------------- #
# Rows are not independent: 1,192+ patients have repeat visits (see EDA), so a
# plain random split would leak the same patient into both train and test.
# A StratifiedGroupKFold is used instead, grouped by PATIENT_ID_COLUMN and
# stratified by CLASSIFICATION_TARGET_COLUMN; one fold is held out as the
# test set. N_SPLITS_FOR_HOLDOUT folds => a 1/N_SPLITS_FOR_HOLDOUT test size.
N_SPLITS_FOR_HOLDOUT: int = 5

TRAIN_DATA_PATH: Path = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH: Path = PROCESSED_DATA_DIR / "test.csv"

# --------------------------------------------------------------------------- #
# Cross-validation (Milestone 9)
# --------------------------------------------------------------------------- #
# Applied *within* the training set only (the held-out test set from
# Milestone 6 is never touched by CV). Folds are grouped by patient id for
# the same reason as the train/test split: repeat-visit rows must not be
# spread across both the CV train and validation portions of a fold.
N_CV_FOLDS: int = 5

# --------------------------------------------------------------------------- #
# Hyperparameter tuning (Milestone 10)
# --------------------------------------------------------------------------- #
# RandomizedSearchCV draws over the group-aware CV splitters from Milestone
# 9; N_TUNING_ITER controls how many parameter combinations are sampled per
# candidate estimator (kept modest -- the dataset is small, so exhaustive
# search is unnecessary and would only slow iteration).
N_TUNING_ITER: int = 20
CLASSIFICATION_TUNING_SCORING: str = "roc_auc"
REGRESSION_TUNING_SCORING: str = "neg_mean_absolute_error"

ARTIFACTS_DIR: Path = PROJECT_ROOT / "ml" / "artifacts"
CLASSIFICATION_TUNING_RESULTS_PATH: Path = ARTIFACTS_DIR / "tuning_results_classification.json"
REGRESSION_TUNING_RESULTS_PATH: Path = ARTIFACTS_DIR / "tuning_results_regression.json"

# --------------------------------------------------------------------------- #
# Model evaluation (Milestone 11)
# --------------------------------------------------------------------------- #
# The held-out test set (Milestone 6) is evaluated exactly once here, using
# the best model/hyperparameters selected via CV in Milestone 10 -- these
# numbers are never fed back into further tuning.
TEST_METRICS_CLASSIFICATION_PATH: Path = ARTIFACTS_DIR / "test_metrics_classification.json"
TEST_METRICS_REGRESSION_PATH: Path = ARTIFACTS_DIR / "test_metrics_regression.json"

# Full model-ladder comparison (incl. the Dummy baseline), each candidate
# refit on the full training set and scored on the held-out test set --
# distinct from CLASSIFICATION_TUNING_RESULTS_PATH, which holds CV-only
# scores used purely for model *selection*, never for reporting.
TEST_LEADERBOARD_CLASSIFICATION_PATH: Path = ARTIFACTS_DIR / "test_leaderboard_classification.json"
TEST_LEADERBOARD_REGRESSION_PATH: Path = ARTIFACTS_DIR / "test_leaderboard_regression.json"

CONFUSION_MATRIX_FIGURE_PATH: Path = REPORT_FIGURES_DIR / "confusion_matrix.png"
ROC_CURVE_FIGURE_PATH: Path = REPORT_FIGURES_DIR / "roc_curve.png"
RESIDUALS_FIGURE_PATH: Path = REPORT_FIGURES_DIR / "residuals.png"

# --------------------------------------------------------------------------- #
# Feature importance & explainability (Milestone 12)
# --------------------------------------------------------------------------- #
FEATURE_IMPORTANCE_CLASSIFICATION_FIGURE_PATH: Path = (
    REPORT_FIGURES_DIR / "feature_importance_classification.png"
)
FEATURE_IMPORTANCE_REGRESSION_FIGURE_PATH: Path = (
    REPORT_FIGURES_DIR / "feature_importance_regression.png"
)
SHAP_SUMMARY_CLASSIFICATION_FIGURE_PATH: Path = REPORT_FIGURES_DIR / "shap_summary_classification.png"
SHAP_SUMMARY_REGRESSION_FIGURE_PATH: Path = REPORT_FIGURES_DIR / "shap_summary_regression.png"
SHAP_MAX_SAMPLES: int = 200

# --------------------------------------------------------------------------- #
# Model persistence (Milestone 13)
# --------------------------------------------------------------------------- #
# The serialized pipelines are what the FastAPI backend (Milestone 14) loads
# at startup -- each preserves its fitted ColumnTransformer alongside the
# estimator, so inference applies identical preprocessing to training.
CLASSIFIER_PIPELINE_PATH: Path = ARTIFACTS_DIR / "classifier_pipeline.joblib"
REGRESSOR_PIPELINE_PATH: Path = ARTIFACTS_DIR / "regressor_pipeline.joblib"
METADATA_PATH: Path = ARTIFACTS_DIR / "metadata.json"
