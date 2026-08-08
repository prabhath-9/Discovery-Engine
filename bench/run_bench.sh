#!/usr/bin/env bash
# 60s headless Locust run against POST /v1/feed, ramping to 200 concurrent users.
# Writes bench/results_stats.csv (+ _stats_history.csv, _failures.csv) and,
# via locustfile.py's request hook, bench/raw_latencies.csv for the histogram.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

HOST="${GATEWAY_URL:-http://localhost:8090}"
USERS="${LOCUST_USERS:-200}"
SPAWN_RATE="${LOCUST_SPAWN_RATE:-20}"
RUN_TIME="${LOCUST_RUN_TIME:-60s}"
OUT_PREFIX="bench/results"

mkdir -p bench

locust -f bench/locustfile.py \
  --headless \
  --host "$HOST" \
  --users "$USERS" \
  --spawn-rate "$SPAWN_RATE" \
  --run-time "$RUN_TIME" \
  --csv "$OUT_PREFIX" \
  --csv-full-history \
  --only-summary

echo
echo "Stats:    ${OUT_PREFIX}_stats.csv"
echo "History:  ${OUT_PREFIX}_stats_history.csv"
echo "Failures: ${OUT_PREFIX}_failures.csv"
echo "Raw:      bench/raw_latencies.csv"
