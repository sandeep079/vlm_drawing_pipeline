#!/bin/bash

# If a specific PDF path is provided, process only that file
if [ -n "$1" ]; then
    python3 src/localize.py "$1"
else
    # Default: Loop through all PDFs in reference_samples/
    echo "No PDF specified. Processing all PDFs in reference_samples/..."
    for pdf in reference_samples/*.pdf; do
        if [ -f "$pdf" ]; then
            echo "========================================"
            echo "Localizing: $pdf"
            echo "========================================"
            python3 src/localize.py "$pdf"
        fi
    done
fi





# for single file execution 
# ./scripts/run_localize.sh reference_samples/001_redacted.pdf