#!/usr/bin/env bash
# Milestone 16 -- local integration check.
#
# Starts the FastAPI backend, waits for it to become healthy, then exercises
# the full predict flow (classification + regression) plus the error states
# required by Milestone 16: invalid input (422) and CORS (allowed vs.
# disallowed origin). Tears the backend down on exit either way.
#
# Run from the repository root: ./scripts/integration_check.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

HOST="127.0.0.1"
PORT=8199
BASE_URL="http://${HOST}:${PORT}"
ALLOWED_ORIGIN="http://localhost:5173"
DISALLOWED_ORIGIN="http://evil.example.com"
LOG_FILE="$(mktemp)"

FAILURES=0

pass() { echo "  PASS: $1"; }
fail() { echo "  FAIL: $1"; FAILURES=$((FAILURES + 1)); }

echo "Starting backend on ${BASE_URL} (log: ${LOG_FILE})..."
python3 -m uvicorn backend.app.main:app --host "$HOST" --port "$PORT" >"$LOG_FILE" 2>&1 &
SERVER_PID=$!

cleanup() {
  echo "Stopping backend (pid ${SERVER_PID})..."
  kill "$SERVER_PID" >/dev/null 2>&1
  wait "$SERVER_PID" 2>/dev/null
}
trap cleanup EXIT

echo "Waiting for /api/v1/health..."
for _ in $(seq 1 30); do
  if curl -s -o /dev/null "${BASE_URL}/api/v1/health"; then
    break
  fi
  sleep 0.5
done

if ! curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}/api/v1/health" | grep -q "200"; then
  echo "Backend never became healthy; last log lines:"
  tail -n 30 "$LOG_FILE"
  exit 1
fi

VALID_PAYLOAD='{"visit_number":3,"is_working_day":true,"has_primary_cancer":false,"has_secondary_cancer":false,"month":"January","day_of_week":"Wednesday","session":"morning","gender":"F","address":"In the city"}'
INVALID_PAYLOAD='{"visit_number":0,"is_working_day":true,"has_primary_cancer":false,"has_secondary_cancer":false,"month":"January","day_of_week":"Wednesday","session":"morning","gender":"F","address":"In the city"}'

echo "1. Health check"
status=$(curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}/api/v1/health")
[ "$status" = "200" ] && pass "GET /api/v1/health -> 200" || fail "GET /api/v1/health -> $status"

echo "2. OpenAPI docs"
status=$(curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}/docs")
[ "$status" = "200" ] && pass "GET /docs -> 200" || fail "GET /docs -> $status"

echo "3. Classification predict (valid payload)"
response=$(curl -s -X POST "${BASE_URL}/api/v1/predict/classification" -H "Content-Type: application/json" -d "$VALID_PAYLOAD")
echo "$response" | grep -q "is_long_consultation" && pass "classification response has expected field" \
  || fail "classification response missing expected field: $response"

echo "4. Regression predict (valid payload)"
response=$(curl -s -X POST "${BASE_URL}/api/v1/predict/regression" -H "Content-Type: application/json" -d "$VALID_PAYLOAD")
echo "$response" | grep -q "predicted_duration_minutes" && pass "regression response has expected field" \
  || fail "regression response missing expected field: $response"

echo "5. Classification predict (invalid payload -> 422)"
status=$(curl -s -o /dev/null -w "%{http_code}" -X POST "${BASE_URL}/api/v1/predict/classification" -H "Content-Type: application/json" -d "$INVALID_PAYLOAD")
[ "$status" = "422" ] && pass "invalid visit_number -> 422" || fail "invalid visit_number -> $status"

echo "6. CORS: allowed origin gets Access-Control-Allow-Origin"
headers=$(curl -s -i -X OPTIONS "${BASE_URL}/api/v1/predict/classification" \
  -H "Origin: ${ALLOWED_ORIGIN}" \
  -H "Access-Control-Request-Method: POST" \
  -H "Access-Control-Request-Headers: Content-Type")
echo "$headers" | grep -qi "access-control-allow-origin: ${ALLOWED_ORIGIN}" \
  && pass "allowed origin (${ALLOWED_ORIGIN}) receives CORS header" \
  || fail "allowed origin (${ALLOWED_ORIGIN}) missing CORS header"

echo "7. CORS: disallowed origin does not get Access-Control-Allow-Origin"
headers=$(curl -s -i -X OPTIONS "${BASE_URL}/api/v1/predict/classification" \
  -H "Origin: ${DISALLOWED_ORIGIN}" \
  -H "Access-Control-Request-Method: POST" \
  -H "Access-Control-Request-Headers: Content-Type")
echo "$headers" | grep -qi "access-control-allow-origin" \
  && fail "disallowed origin (${DISALLOWED_ORIGIN}) unexpectedly received a CORS header" \
  || pass "disallowed origin (${DISALLOWED_ORIGIN}) correctly receives no CORS header"

echo
if [ "$FAILURES" -eq 0 ]; then
  echo "All integration checks passed."
  exit 0
else
  echo "${FAILURES} integration check(s) failed."
  exit 1
fi
