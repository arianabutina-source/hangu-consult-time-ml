// Mirrors the JSON shapes written by scripts/export_dashboard_data.py.

export interface ConfusionMatrixData {
  labels: [string, string];
  matrix: [[number, number], [number, number]];
}

export interface RocCurvePoint {
  fpr: number;
  tpr: number;
}

export interface FeatureImportanceRecord {
  feature: string;
  importance: number;
}

export interface ResidualPoint {
  predicted: number;
  residual: number;
}

export interface TestMetricsClassification {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
}

export interface TestMetricsRegression {
  mae: number;
  rmse: number;
  r2: number;
}
