#!/usr/bin/env bash
#
# Quick smoke test: run lobster_hmf on 5 AIME samples
# to verify everything works end-to-end.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

echo "=== HMF Quick Smoke Test ==="
echo "Running lobster_hmf on aime_2025 (5 samples)..."
echo ""

bash scripts/run_hmf_benchmark.sh lobster_hmf aime_2025 --limit 5

echo ""
echo "=== Smoke test complete ==="
