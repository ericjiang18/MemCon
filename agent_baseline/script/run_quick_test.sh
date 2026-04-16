#!/usr/bin/env bash
# Quick smoke test – runs 3 samples per benchmark on every framework.
# Usage:  bash script/run_quick_test.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
bash "$SCRIPT_DIR/run_all.sh" --limit 3
