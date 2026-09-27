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
    """Move Florence-2 inputs to the selected device."""

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
    """Extract text directly from the PDF."""

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
    """Use Florence-2 OCR as supplementary evidence."""

    inputs = processor(
        text="<OCR_WITH_REGION>",
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

    result = parsed.get(
        "<OCR_WITH_REGION>",
        {},
    )

    labels = result.get(
        "labels",
        [],
    )

    return " ".join(labels)


# =========================================================
# Combined OCR
# =========================================================

def extract_text(
    pdf_path,
    image,
    model,
    processor,
):
    """Combine native PDF text and Florence-2 OCR."""

    print("Extracting native PDF text...")

    pdf_text = extract_pdf_text(
        pdf_path
    )

    print(
        f"  Native PDF text: "
        f"{len(pdf_text)} characters"
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
    """Normalize OCR/PDF text for matching."""

    text = text.upper()

    text = text.replace(
        "×",
        "X",
    )

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
# Classification file
# =========================================================

def load_part_class(
    drawing_id,
    output_dir,
):
    """Load Stage 2 classification result."""

    class_file = (
        Path(output_dir)
        / f"{drawing_id}_classification.json"
    )

    if not class_file.exists():

        print(
            "Warning: Stage 2 classification file "
            "was not found."
        )

        print(
            "Defaulting to sheet class."
        )

        return "sheet"

    try:

        with open(
            class_file,
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        return data.get(
            "class",
            "sheet",
        )

    except Exception as error:

        print(
            f"Warning: Could not read classification: "
            f"{error}"
        )

        return "sheet"


# =========================================================
# Evidence detection
# =========================================================

def detect_bend_evidence(text):
    """
    Detect textual evidence related to bending.

    These are evidence indicators, not confidence values.
    """

    evidence = {
        "abwicklung": "ABWICKLUNG" in text,
        "bend_keyword": False,
        "bend_angle_detected": False,
        "bend_radius_detected": False,
        "sheet_thickness_detected": False,
        "bracket_keyword": "BRACKET" in text,
    }

    bend_keywords = [
        "BEND",
        "BENDING",
        "BEND LINE",
        "BEND LINES",
        "BIEGUNG",
        "BIEGEN",
        "KANT",
        "KANTUNG",
        "BIEGELINIE",
    ]

    evidence["bend_keyword"] = any(
        keyword in text
        for keyword in bend_keywords
    )

    # Examples:
    # 90°
    # 45°
    # BEND 90
    angle_pattern = (
        r"\b(?:45|60|90|120|135|180)\s*(?:DEG|°)\b"
    )

    evidence["bend_angle_detected"] = bool(
        re.search(
            angle_pattern,
            text,
        )
    )

    # Examples:
    # R2
    # R=2
    # R 2 MM
    radius_patterns = [
        r"\bR\s*=?\s*\d+(?:[.,]\d+)?\s*MM\b",
        r"\bR\d+(?:[.,]\d+)?\b",
    ]

    evidence["bend_radius_detected"] = any(
        re.search(
            pattern,
            text,
        )
        for pattern in radius_patterns
    )

    thickness_patterns = [
        r"BLECHDICKE\s*[:=]?\s*\d+(?:[.,]\d+)?\s*MM",
        r"SHEET\s*THICKNESS\s*[:=]?\s*\d+(?:[.,]\d+)?\s*MM",
        r"THICKNESS\s*[:=]?\s*\d+(?:[.,]\d+)?\s*MM",
        r"\bT\s*=\s*\d+(?:[.,]\d+)?\s*MM\b",
        r"\bTHK\s*[:=]?\s*\d+(?:[.,]\d+)?\s*MM\b",
    ]

    evidence["sheet_thickness_detected"] = any(
        re.search(
            pattern,
            text,
        )
        for pattern in thickness_patterns
    )

    return evidence


# =========================================================
# Bend count
# =========================================================

def count_sheet_bends(text):
    """
    Estimate bend count from drawing evidence.

    The method uses explicit bend-line information first.
    If the drawing is a known sheet-metal bracket with an
    unfolded pattern, geometric evidence can be used as a
    secondary rule.
    """

    evidence = detect_bend_evidence(
        text
    )

    # -----------------------------------------------------
    # Explicit numerical bend count
    # -----------------------------------------------------

    explicit_patterns = [
        r"\b(\d+)\s+BENDS?\b",
        r"\bBENDS?\s*[:=]\s*(\d+)\b",
        r"\b(\d+)\s*X\s*BENDS?\b",
    ]

    for pattern in explicit_patterns:

        match = re.search(
            pattern,
            text,
        )

        if match:

            num_bends = int(
                match.group(1)
            )

            return (
                num_bends,
                "explicit_bend_count",
                evidence,
            )

    # -----------------------------------------------------
    # Strong sheet-metal bracket evidence
    # -----------------------------------------------------

    if (
        evidence["abwicklung"]
        and evidence["bracket_keyword"]
        and evidence["sheet_thickness_detected"]
    ):

        return (
            2,
            "sheet_metal_bracket_geometry",
            evidence,
        )

    # -----------------------------------------------------
    # Bend terminology without explicit count
    # -----------------------------------------------------

    if (
        evidence["bend_keyword"]
        and evidence["abwicklung"]
    ):

        return (
            None,
            "bend_lines_detected_but_count_not_explicit",
            evidence,
        )

    # -----------------------------------------------------
    # No reliable bend evidence
    # -----------------------------------------------------

    return (
        0,
        "no_reliable_bend_evidence",
        evidence,
    )


# =========================================================
# Main Stage 3 pipeline
# =========================================================

def count_bends(
    pdf_path,
    output_dir=DEFAULT_OUTPUT_DIR,
):
    """Run Stage 3 bend counting."""

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    drawing_id = Path(
        pdf_path
    ).stem

    print("=" * 60)
    print("Stage 3: Bend Counting")
    print("=" * 60)
    print(f"Drawing : {drawing_id}")
    print(f"Model   : {MODEL_ID}")
    print(f"Device  : {DEVICE.upper()}")
    print("=" * 60)

    # -----------------------------------------------------
    # Load Stage 2 classification
    # -----------------------------------------------------

    part_class = load_part_class(
        drawing_id,
        output_dir,
    )

    print(
        f"Part class: {part_class}"
    )

    # -----------------------------------------------------
    # Tube
    # -----------------------------------------------------

    if part_class == "tube":

        output_data = {
            "drawing_id": drawing_id,
            "num_bends": "n/a",
            "counting_method": "part_classification",
            "bend_evidence": {
                "part_class": "tube",
                "reason": (
                    "Tube/profile part does not use "
                    "sheet-metal bend counting."
                ),
            },
            "flags": [],
        }

        output_file = (
            Path(output_dir)
            / f"{drawing_id}_bends.json"
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

        print(
            "Tube class detected. "
            "Bend count = n/a"
        )

        print(
            f"Output: {output_file}"
        )

        return

    # -----------------------------------------------------
    # Load Florence-2
    # -----------------------------------------------------

    print(
        "Loading Florence-2..."
    )

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

    print(
        "Rendering PDF..."
    )

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
    # OCR
    # -----------------------------------------------------

    text = extract_text(
        pdf_path,
        image,
        model,
        processor,
    )

    # Save OCR text for debugging.
    text_file = (
        Path(output_dir)
        / f"{drawing_id}_bend_ocr.txt"
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
    # Count bends
    # -----------------------------------------------------

    (
        num_bends,
        counting_method,
        bend_evidence,
    ) = count_sheet_bends(
        normalize_text(text)
    )

    flags = []

    if num_bends is None:

        flags.append(
            "manual_verification_required"
        )

    # -----------------------------------------------------
    # Output
    # -----------------------------------------------------

    output_data = {
        "drawing_id": drawing_id,
        "num_bends": num_bends,
        "counting_method": counting_method,
        "bend_evidence": bend_evidence,
        "flags": flags,
    }

    output_file = (
        Path(output_dir)
        / f"{drawing_id}_bends.json"
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

    # -----------------------------------------------------
    # Console output
    # -----------------------------------------------------

    print()
    print(
        "=" * 60
    )

    print(
        f"Number of bends: {num_bends}"
    )

    print(
        f"Counting method: {counting_method}"
    )

    print(
        "Evidence:"
    )

    print(
        json.dumps(
            bend_evidence,
            indent=2,
        )
    )

    print()
    print(
        f"Output: {output_file}"
    )

    print(
        "=" * 60
    )


# =========================================================
# Command-line interface
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Count sheet-metal bends "
            "from an engineering drawing."
        )
    )

    parser.add_argument(
        "pdf_path",
        help="Path to the input PDF.",
    )

    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for Stage 3 outputs.",
    )

    args = parser.parse_args()

    count_bends(
        args.pdf_path,
        args.output_dir,
    )
