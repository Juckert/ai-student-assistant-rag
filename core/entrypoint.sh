#!/bin/sh
set -e

echo ""
echo "========================================"
echo "  Running test suite"
echo "========================================"
python -m pytest tests/ -v --tb=short
echo "========================================"
echo "  All tests passed — starting assistant"
echo "========================================"
echo ""

exec python -m streamlit run app/app.py \
    --server.headless true \
    --server.port 8501 \
    --server.address 0.0.0.0
