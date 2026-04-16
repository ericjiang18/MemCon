#!/usr/bin/env bash
# install_deps.sh – Install all Python & Node dependencies needed for benchmarking.
#
# Usage:  bash script/install_deps.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Use "python3 -m pip" so packages land in the SAME Python that scripts invoke.
PIP="python3 -m pip"

echo "=========================================="
echo "  Python: $(python3 --version)  ($(which python3))"
echo "=========================================="

echo ""
echo "=========================================="
echo "  Installing base Python dependencies"
echo "=========================================="
$PIP install --quiet openai python-dotenv tqdm class-registry datasets huggingface_hub \
    langchain-openai langchain-core

echo ""
echo "=========================================="
echo "  Installing agent-framework (from source)"
echo "=========================================="
$PIP install --quiet -e "$PROJECT_DIR/agent-framework/python/packages/core" \
    || echo "  [WARN] agent-framework install failed – check $PROJECT_DIR/agent-framework/python/packages/core"

echo ""
echo "=========================================="
echo "  Installing AutoGen (from source)"
echo "=========================================="
$PIP install --quiet -e "$PROJECT_DIR/autogen/python/packages/autogen-core" \
    || echo "  [WARN] autogen-core install failed"
$PIP install --quiet -e "$PROJECT_DIR/autogen/python/packages/autogen-agentchat" \
    || echo "  [WARN] autogen-agentchat install failed"
$PIP install --quiet -e "$PROJECT_DIR/autogen/python/packages/autogen-ext[openai]" \
    || echo "  [WARN] autogen-ext install failed"

echo ""
echo "=========================================="
echo "  Installing LangGraph (from source)"
echo "=========================================="
$PIP install --quiet -e "$PROJECT_DIR/langgraph/libs/langgraph" \
    || echo "  [WARN] langgraph install failed"
$PIP install --quiet -e "$PROJECT_DIR/langgraph/libs/prebuilt" \
    || echo "  [WARN] langgraph-prebuilt install failed"

echo ""
echo "=========================================="
echo "  Setting up Lobster (Node.js)"
echo "=========================================="
if command -v node &>/dev/null; then
    echo "  Node.js $(node --version) found"
    cd "$PROJECT_DIR/lobster"
    if [ -f package.json ]; then
        npm install --silent 2>/dev/null || echo "  [WARN] npm install failed in lobster/"
    fi
    cd "$PROJECT_DIR"
else
    echo "  [WARN] Node.js not found – Lobster will fall back to Python openai client"
fi

echo ""
echo "=========================================="
echo "  Verifying core imports …"
echo "=========================================="
python3 -c "
import datasets, openai, class_registry
print('  datasets   :', datasets.__version__)
print('  openai     :', openai.__version__)
print('  OK – core imports working')
" || echo "  [ERROR] Core import check failed – run:  python3 -m pip install datasets openai class-registry"

echo ""
echo "=========================================="
echo "  All done!"
echo "=========================================="
