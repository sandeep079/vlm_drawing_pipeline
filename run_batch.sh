#!/usr/bin/env bash
set -e

for pdf in reference_samples/*.pdf; do
    echo "Processing $pdf with VLM pipeline..."
    python3 src/localize.py "$pdf"
    python3 src/classify.py "$pdf"
    python3 src/count_bends.py "$pdf"
done
