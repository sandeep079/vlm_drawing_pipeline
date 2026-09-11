#!/usr/bin/env bash

# Stop execution if any command fails
set -e

# Set target directory (defaults to reference_samples if no argument is passed)
TARGET_DIR="${1:-reference_samples}"

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory '$TARGET_DIR' does not exist."
    exit 1
fi

echo "Starting batch processing for PDFs in '$TARGET_DIR'..."

# Enable nullglob to safely handle cases where no .pdf files exist
shopt -s nullglob
pdf_files=("$TARGET_DIR"/*.pdf)
shopt -u nullglob

if [ ${#pdf_files[@]} -eq 0 ]; then
    echo "No PDF files found in '$TARGET_DIR'."
    exit 0
fi

for file in "${pdf_files[@]}"; do
    echo "========================================"
    echo "Processing: $file"
    echo "========================================"
    python vlm_drawing_pipeline.py "$file"
done

echo "Batch processing finished successfully."
