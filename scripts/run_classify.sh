#!/bin/bash

if [ -n "$1" ]; then
    python3 src/classify.py "$1"
else
    echo "No PDF specified. Processing all PDFs in reference_samples/..."
    for pdf in reference_samples/*.pdf; do
        if [ -f "$pdf" ]; then
            echo "========================================"
            echo "Classifying: $pdf"
            echo "========================================"
            python3 src/classify.py "$pdf"
        fi
    done
fi