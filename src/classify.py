import argparse
import json
import os
import re
from pathlib import Path

import pdfplumber
import torch
from pdf2image import convert_from_path
from transformers import AutoModelForCausalLM, AutoProcessor


# =========================================================
# Configuration
# =========================================================

MODEL_ID = "microsoft/Florence-2-base"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_OUTPUT_DIR = "output"


# =========================================================
# Input preparation
# =========================================================

def prepare_inputs(inputs):
    """Move processor inputs to the selected device."""

    prepared = {}

    for key, value in inputs.items():

        if DEVICE == "cuda" and value.dtype == torch.float32:
            prepared[key] = value.to(
                DEVICE,
                dtype=torch.float16,
            )
        else:
            prepared[key] = value.to(DEVICE)

    return prepared


# =========================================================
# Native PDF text extraction
# =========================================================

def extract_pdf_text(pdf_path):
    """
    Extract text directly from the PDF.

    This is usually faster and cleaner than OCR when the PDF
    contains actual text objects.
    """

    text_parts = []

    try:

        with pdfplumber.open(pdf_path) as pdf:

            for page in pdf.pages:

                page_text = page.extract_text()

                if page_text:
                    text_parts.append(page_text)

    except Exception as error:

        print(
            f"Warning: PDF text extraction failed: {error}"
        )

    return "\n".join(text_parts)


# =========================================================
# Florence-2 OCR
# =========================================================

def extract_vlm_text(
    image,
    model,
    processor,
):
    """
    Use Florence-2 OCR as a supplementary text source.

    Native PDF text extraction is preferred when available.
    """

    prompt = "<OCR_WITH_REGION>"

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
    )

    inputs = prepare_inputs(inputs)

    with torch.no_grad():

        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=512,
            num_beams=3,
        )

    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=False,
    )[0]

    parsed = processor.post_process_generation(
        generated_text,
        task="<OCR_WITH_REGION>",
        image_size=image.size,
    )

    ocr_result = parsed.get(
        "<OCR_WITH_REGION>",
        {},
    )

    labels = ocr_result.get(
        "labels",
        [],
    )

    return " ".join(labels)


# =========================================================
# Combined text extraction
# =========================================================

def extract_text(
    pdf_path,
    image,
    model,
    processor,
):
    """
    Combine native PDF text and Florence-2 OCR.

    The combined text is normalized to uppercase so that
    keyword matching is case-insensitive.
    """

    print("Extracting native PDF text...")

    pdf_text = extract_pdf_text(
        pdf_path
    )

    if pdf_text.strip():

        print(
            f"  Native PDF text: "
            f"{len(pdf_text)} characters"
        )

    else:

        print(
            "  No usable native PDF text found."
        )

    print("Running Florence-2 OCR...")

    try:

        vlm_text = extract_vlm_text(
            image,
            model,
            processor,
        )

        print(
            f"  VLM OCR text: "
            f"{len(vlm_text)} characters"
        )

    except Exception as error:

        print(
            f"Warning: VLM OCR failed: {error}"
        )

        vlm_text = ""

    combined_text = (
        pdf_text
        + "\n"
        + vlm_text
    ).upper()

    return combined_text


# =========================================================
# Text normalization
# =========================================================

def normalize_text(text):
    """
    Normalize common OCR/PDF extraction variations.

    This helps keyword matching when OCR introduces spaces,
    punctuation, or formatting differences.
    """

    text = text.upper()

    # Normalize multiplication signs.
    text = text.replace("×", "X")

    # Replace common separators with spaces.
    text = re.sub(
        r"[\t\r\n]+",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


# =========================================================
# Keyword detection
# =========================================================

SHEET_KEYWORDS = [
    "ABWICKLUNG",
    "BLECH",
    "BLECHDICKE",
    "SHEET",
    "SHEET METAL",
    "BRACKET",
    "PLATE",
    "ADAPTER",
    "BEND",
    "BENDING",
    "KANT",
    "KANTUNG",
]


TUBE_KEYWORDS = [
    "SQUARE TUBE",
    "RECTANGULAR TUBE",
    "HOLLOW SECTION",
    "HOLLOW PROFILE",
    "TUBE",
    "ROHR",
    "4-KANT",
    "4KANT",
    "QUADRATROHR",
    "PROFILE",
]


# Material / thickness indicators frequently found on
# sheet-metal engineering drawings.
SHEET_MATERIAL_KEYWORDS = [
    "ALMG3",
    "1.4301",
    "1.0038",
    "STAINLESS",
    "ALUMINIUM",
    "ALUMINUM",
    "STEEL",
]


def find_keywords(
    text,
    keywords,
):
    """Return all matching keywords."""

    found = []

    for keyword in keywords:

        if keyword in text:
            found.append(keyword)

    return found


def has_sheet_thickness_pattern(text):
    """
    Detect common sheet-thickness notation.

    Examples:
        BLECHDICKE 2 MM
        2 MM
        T=2
        THK 2
        THICKNESS 2
    """

    patterns = [
        r"BLECHDICKE\s*[:=]?\s*\d+(?:[.,]\d+)?\s*MM",
        r"SHEET\s*THICKNESS\s*[:=]?\s*\d+(?:[.,]\d+)?\s*MM",
        r"THICKNESS\s*[:=]?\s*\d+(?:[.,]\d+)?\s*MM",
        r"\bT\s*=\s*\d+(?:[.,]\d+)?\s*MM\b",
        r"\bTHK\s*[:=]?\s*\d+(?:[.,]\d+)?\s*MM\b",
    ]

    return any(
        re.search(pattern, text)
        for pattern in patterns
    )


def has_tube_dimension_pattern(text):
    """
    Detect common tube/profile dimensions.

    Examples:
        80X80
        80 X 80 X 3
        100X50
    """

    patterns = [
        r"\b\d{2,4}\s*[X]\s*\d{2,4}\b",
        r"\b\d{2,4}\s*[X]\s*\d{2,4}\s*[X]\s*\d+(?:[.,]\d+)?\b",
    ]

    return any(
        re.search(pattern, text)
        for pattern in patterns
    )


# =========================================================
# Classification
# =========================================================

def classify_text(text):
    """
    Classify the drawing as sheet or tube using textual evidence.

    The method is intentionally transparent rather than pretending
    that the hard-coded evidence weights are a machine-learning
    confidence score.
    """

    text = normalize_text(text)

    sheet_keywords = find_keywords(
        text,
        SHEET_KEYWORDS,
    )

    tube_keywords = find_keywords(
        text,
        TUBE_KEYWORDS,
    )

    material_keywords = find_keywords(
        text,
        SHEET_MATERIAL_KEYWORDS,
    )

    sheet_thickness = has_sheet_thickness_pattern(
        text
    )

    tube_dimensions = has_tube_dimension_pattern(
        text
    )

    # -----------------------------------------------------
    # Evidence scores
    # -----------------------------------------------------

    sheet_score = 0
    tube_score = 0

    sheet_score += len(sheet_keywords) * 2
    tube_score += len(tube_keywords) * 2

    if material_keywords:
        sheet_score += 1

    if sheet_thickness:
        sheet_score += 3

    if tube_dimensions:
        tube_score += 2

    # Explicit tube terminology is stronger than a generic
    # dimensional pattern.
    if tube_keywords:
        tube_score += 2

    # Explicit sheet-metal terminology is strong evidence.
    if "ABWICKLUNG" in text:
        sheet_score += 4

    if "BLECHDICKE" in text:
        sheet_score += 4

    if "BRACKET" in text:
        sheet_score += 3

    if "PLATE" in text:
        sheet_score += 2

    # -----------------------------------------------------
    # Determine class
    # -----------------------------------------------------

    if tube_score > sheet_score:
        part_class = "tube"

    else:
        part_class = "sheet"

    # -----------------------------------------------------
    # Build evidence description
    # -----------------------------------------------------

    evidence = {
        "sheet_keywords": sheet_keywords,
        "tube_keywords": tube_keywords,
        "material_keywords": material_keywords,
        "sheet_thickness_detected": sheet_thickness,
        "tube_dimension_detected": tube_dimensions,
        "sheet_evidence_score": sheet_score,
        "tube_evidence_score": tube_score,
    }

    return part_class, evidence


# =========================================================
# Main Stage 2 pipeline
# =========================================================

def classify_drawing(
    pdf_path,
    output_dir=DEFAULT_OUTPUT_DIR,
):
    """
    Run Stage 2 drawing classification.
    """

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    drawing_id = Path(
        pdf_path
    ).stem

    print("=" * 60)
    print("Stage 2: Drawing Classification")
    print("=" * 60)
    print(f"Drawing : {drawing_id}")
    print(f"Model   : {MODEL_ID}")
    print(f"Device  : {DEVICE.upper()}")
    print("=" * 60)

    # -----------------------------------------------------
    # Load Florence-2
    # -----------------------------------------------------

    print("Loading Florence-2...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=(
            torch.float16
            if DEVICE == "cuda"
            else torch.float32
        ),
        trust_remote_code=True,
    ).to(DEVICE)

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )

    model.eval()

    print(
        "Florence-2 loaded successfully."
    )

    # -----------------------------------------------------
    # Render PDF
    # -----------------------------------------------------

    print("Rendering PDF...")

    pages = convert_from_path(
        pdf_path,
        dpi=300,
    )

    if not pages:
        raise RuntimeError(
            "No pages were found in the PDF."
        )

    image = pages[0]

    print(
        f"Page size: "
        f"{image.width} x {image.height}"
    )

    # -----------------------------------------------------
    # Extract text
    # -----------------------------------------------------

    text = extract_text(
        pdf_path,
        image,
        model,
        processor,
    )

    # Save extracted text for debugging/reproducibility.
    text_file = (
        Path(output_dir)
        / f"{drawing_id}_ocr.txt"
    )

    with open(
        text_file,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(text)

    print(
        f"OCR text saved to: {text_file}"
    )

    # -----------------------------------------------------
    # Classify
    # -----------------------------------------------------

    part_class, evidence = classify_text(
        text
    )

    print()
    print(
        f"Classification: {part_class}"
    )

    print(
        "Evidence:"
    )

    print(
        json.dumps(
            evidence,
            indent=2,
        )
    )

    # -----------------------------------------------------
    # Save result
    # -----------------------------------------------------

    output_data = {
        "drawing_id": drawing_id,
        "class": part_class,
        "classification_method": (
            "native_pdf_text_plus_florence2_ocr"
        ),
        "class_evidence": evidence,
    }

    output_file = (
        Path(output_dir)
        / f"{drawing_id}_classification.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output_data,
            file,
            indent=2,
        )

    print()
    print("=" * 60)
    print("Classification completed.")
    print("=" * 60)
    print(
        f"Output: {output_file}"
    )

    print(
        json.dumps(
            output_data,
            indent=2,
        )
    )


# =========================================================
# Command-line interface
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Classify an engineering drawing "
            "as sheet or tube."
        )
    )

    parser.add_argument(
        "pdf_path",
        help="Path to the input manufacturing drawing PDF.",
    )

    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for Stage 2 outputs.",
    )

    args = parser.parse_args()

    classify_drawing(
        args.pdf_path,
        args.output_dir,
    )