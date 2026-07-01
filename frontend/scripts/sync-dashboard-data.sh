#!/usr/bin/env bash
# Copies the static dashboard JSON exported by
# `python -m scripts.export_dashboard_data` (see ../../scripts) into
# public/data/, so the dashboard has no runtime dependency on the ml/
# package or a live backend. Re-run after retraining/re-tuning the models.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$FRONTEND_DIR")"
ARTIFACTS_DIR="$REPO_ROOT/ml/artifacts"
DEST_DIR="$FRONTEND_DIR/public/data"

mkdir -p "$DEST_DIR"
cp "$ARTIFACTS_DIR/dashboard/"*.json "$DEST_DIR/"
cp "$ARTIFACTS_DIR/test_metrics_classification.json" "$DEST_DIR/"
cp "$ARTIFACTS_DIR/test_metrics_regression.json" "$DEST_DIR/"

echo "Synced dashboard data into $DEST_DIR"
