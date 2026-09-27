import argparse
import json
import os
from pathlib import Path

import torch
from PIL import ImageDraw
from pdf2image import convert_from_path
from transformers import AutoModelForCausalLM, AutoProcessor



MODEL_ID = "microsoft/Florence-2-base"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"      

OUTPUT_DIR = "output"

# Required for stage 1 localization
TARGET_LABELS = [
    "flat_pattern",
    "projection_view",
    "isometric_view",
]

# Florence-2 understands natural language better than internal
GROUNDING_PHRASES = {
    "flat_pattern": [
        "unrolled flat sheet metal pattern",
        "flat pattern of the sheet metal part",
        "unfolded sheet metal pattern",
    ],
    "projection_view": [
        "orthographic projection views of the part",
        "orthographic drawing of the part",
        "front top side views of the part",
    ],
    "isometric_view": [
        "3D isometric view of the part",
        "isometric drawing of the part",
        "three dimensional pictorial view of the part",
    ],
}


# =========================================================
# Detection parameters
# =========================================================

# Ignore extremely small boxes.
MIN_BOX_AREA = 2_000

# Reject boxes that occupy almost the complete page.
MAX_PAGE_AREA_RATIO = 0.65

# If two boxes overlap this much, they are probably duplicate
# detections of the same drawing region.
MAX_OVERLAP_RATIO = 0.70


# =========================================================
# Title block geometry
# =========================================================
#
# Engineering drawings normally place the title block in the
# lower-right area of the sheet.
#
# These values define the SEARCH AREA, not the final title box.
# The image-processing step below searches this area for the
# actual table-like region.
#

TITLE_SEARCH_X = 0.55
TITLE_SEARCH_Y = 0.65


# =========================================================
# Helper functions
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


def clamp_bbox(bbox, width, height):
    """Clamp a bounding box to the image boundaries."""

    if len(bbox) != 4:
        return None

    x1, y1, x2, y2 = [int(v) for v in bbox]

    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


def bbox_area(bbox):
    """Return bounding-box area."""

    x1, y1, x2, y2 = bbox

    return max(0, x2 - x1) * max(0, y2 - y1)


def intersection_area(box_a, box_b):
    """Return intersection area between two bounding boxes."""

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    if x2 <= x1 or y2 <= y1:
        return 0

    return (x2 - x1) * (y2 - y1)


def overlap_ratio(box_a, box_b):
    """Return intersection relative to the smaller box."""

    area_a = bbox_area(box_a)
    area_b = bbox_area(box_b)

    if area_a == 0 or area_b == 0:
        return 0.0

    intersection = intersection_area(box_a, box_b)

    return intersection / min(area_a, area_b)


def is_reasonable_vlm_box(bbox, width, height):
    """Reject obviously bad Florence-2 detections."""

    area = bbox_area(bbox)
    page_area = width * height

    if area < MIN_BOX_AREA:
        return False

    if area / page_area > MAX_PAGE_AREA_RATIO:
        return False

    return True


# =========================================================
# Florence-2 localization
# =========================================================

def run_florence_grounding(
    model,
    processor,
    image,
    phrase,
):
    """Run Florence-2 phrase grounding for one phrase."""

    prompt = (
        "<CAPTION_TO_PHRASE_GROUNDING>"
        + phrase
    )

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
            max_new_tokens=256,
            num_beams=3,
        )

    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=False,
    )[0]

    result = processor.post_process_generation(
        generated_text,
        task="<CAPTION_TO_PHRASE_GROUNDING>",
        image_size=image.size,
    )

    return result.get(
        "<CAPTION_TO_PHRASE_GROUNDING>",
        {},
    )


def localize_with_florence(
    model,
    processor,
    image,
    label,
):
    """
    Try several natural-language descriptions for one region.

    Florence-2 does not provide a reliable confidence score for
    these detections, so we keep candidate boxes and later apply
    geometric validation.
    """

    width, height = image.size

    candidates = []

    for phrase in GROUNDING_PHRASES[label]:

        print(f"    Prompt: {phrase}")

        grounding = run_florence_grounding(
            model,
            processor,
            image,
            phrase,
        )

        boxes = grounding.get(
            "bboxes",
            [],
        )

        for box in boxes:

            box = clamp_bbox(
                box,
                width,
                height,
            )

            if box is None:
                continue

            if not is_reasonable_vlm_box(
                box,
                width,
                height,
            ):
                print(
                    f"      Rejected box: {box}"
                )
                continue

            candidates.append(
                {
                    "label": label,
                    "box_2d": box,
                    "source": "florence2",
                    "prompt": phrase,
                }
            )

            print(
                f"      Candidate: {box}"
            )

    return candidates


# =========================================================
# Candidate selection
# =========================================================

def select_best_candidate(candidates):
    """
    Select a representative candidate.

    We prefer the smaller box when detections overlap heavily,
    because Florence-2 sometimes returns a large region containing
    the actual target.
    """

    if not candidates:
        return None

    candidates = sorted(
        candidates,
        key=lambda item: bbox_area(
            item["box_2d"]
        ),
    )

    selected = candidates[0]

    return {
        "label": selected["label"],
        "box_2d": selected["box_2d"],
        "source": selected["source"],
    }


def remove_duplicate_regions(regions):
    """
    Remove heavily overlapping detections belonging to different
    labels.

    This prevents the same large Florence-2 box from becoming both
    flat_pattern and projection_view.
    """

    final_regions = []

    # Smaller boxes are processed first.
    regions = sorted(
        regions,
        key=lambda item: bbox_area(
            item["box_2d"]
        ),
    )

    for region in regions:

        duplicate = False

        for existing in final_regions:

            overlap = overlap_ratio(
                region["box_2d"],
                existing["box_2d"],
            )

            if overlap >= MAX_OVERLAP_RATIO:
                duplicate = True
                break

        if not duplicate:
            final_regions.append(region)

    return final_regions


# =========================================================
# Title block detection
# =========================================================

def detect_title_block(image):
    """
    Detect a title block using the known spatial structure of
    engineering drawings.

    Florence-2 is deliberately NOT used here.

    The function searches the bottom-right part of the drawing
    and looks for dark table/grid structures.
    """

    width, height = image.size

    search_x1 = int(width * TITLE_SEARCH_X)
    search_y1 = int(height * TITLE_SEARCH_Y)

    search_x2 = width
    search_y2 = height

    crop = image.crop(
        (
            search_x1,
            search_y1,
            search_x2,
            search_y2,
        )
    )

    # Convert to grayscale.
    gray = crop.convert("L")

    # Detect dark pixels.
    #
    # Engineering drawing tables contain many dark horizontal
    # and vertical lines.
    threshold = 180

    binary = gray.point(
        lambda pixel: 255
        if pixel < threshold
        else 0
    )

    bbox = binary.getbbox()

    if bbox is None:
        return None

    bx1, by1, bx2, by2 = bbox

    # Convert coordinates back to the full-page image.
    x1 = search_x1 + bx1
    y1 = search_y1 + by1
    x2 = search_x1 + bx2
    y2 = search_y1 + by2

    detected = clamp_bbox(
        [x1, y1, x2, y2],
        width,
        height,
    )

    if detected is None:
        return None

    # Prevent the entire lower-right quadrant from being accepted.
    area_ratio = bbox_area(detected) / (
        width * height
    )

    if area_ratio > 0.25:
        return None

    return detected


def fallback_title_block(width, height):
    """
    Conservative bottom-right fallback.

    This is used only when image-based detection fails.
    """

    return [
        int(width * 0.70),
        int(height * 0.78),
        width,
        height,
    ]


# =========================================================
# Crop generation
# =========================================================

def save_crop(
    image,
    bbox,
    output_path,
):
    """Save one region crop."""

    x1, y1, x2, y2 = bbox

    crop = image.crop(
        (
            x1,
            y1,
            x2,
            y2,
        )
    )

    crop.save(output_path)


# =========================================================
# Visualization
# =========================================================

def save_visualization(
    image,
    regions,
    output_path,
):
    """Draw all detected regions on the original page."""

    visualization = image.copy()

    draw = ImageDraw.Draw(
        visualization
    )

    for region in regions:

        x1, y1, x2, y2 = region["box_2d"]
        label = region["label"]

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=8,
        )

        text_position = (
            x1 + 10,
            max(10, y1 - 35),
        )

        draw.text(
            text_position,
            label,
            fill="red",
        )

    visualization.save(
        output_path
    )


# =========================================================
# Main localization pipeline
# =========================================================

def localize_drawing(
    pdf_path,
    output_dir=OUTPUT_DIR,
):
    """Run Stage 1 localization."""

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    drawing_id = Path(
        pdf_path
    ).stem

    print("=" * 60)
    print("Stage 1: Drawing Region Localization")
    print("=" * 60)
    print(f"Drawing : {drawing_id}")
    print(f"Model   : {MODEL_ID}")
    print(f"Device  : {DEVICE.upper()}")
    print("=" * 60)

    # -----------------------------------------------------
    # Load model
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

    print("Florence-2 loaded successfully.")

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

    width, height = image.size

    print(
        f"Page size: {width} x {height}"
    )

    # -----------------------------------------------------
    # Localize the three view types
    # -----------------------------------------------------

    all_regions = []

    for label in TARGET_LABELS:

        print()
        print(
            f"Searching for: {label}"
        )

        candidates = localize_with_florence(
            model,
            processor,
            image,
            label,
        )

        best = select_best_candidate(
            candidates
        )

        if best is not None:

            all_regions.append(best)

            print(
                f"  Selected {label}: "
                f"{best['box_2d']}"
            )

        else:

            print(
                f"  No reliable {label} "
                f"detection found."
            )

    # -----------------------------------------------------
    # Remove duplicate view boxes
    # -----------------------------------------------------

    regions = remove_duplicate_regions(
        all_regions
    )

    # -----------------------------------------------------
    # Detect title block separately
    # -----------------------------------------------------

    print()
    print("Searching for: title_block")

    title_bbox = detect_title_block(
        image
    )

    if title_bbox is not None:

        print(
            f"  Title block detected: "
            f"{title_bbox}"
        )

        regions.append(
            {
                "label": "title_block",
                "box_2d": title_bbox,
                "source": "layout_detection",
            }
        )

    else:

        print(
            "  Automatic title-block detection "
            "failed."
        )

        title_bbox = fallback_title_block(
            width,
            height,
        )

        regions.append(
            {
                "label": "title_block",
                "box_2d": title_bbox,
                "source": "fallback",
            }
        )

        print(
            f"  Using fallback: {title_bbox}"
        )

    # -----------------------------------------------------
    # Save crops
    # -----------------------------------------------------

    print()
    print("Saving crops...")

    label_counts = {}

    for region in regions:

        label = region["label"]

        count = label_counts.get(
            label,
            0,
        )

        label_counts[label] = count + 1

        filename = (
            f"{drawing_id}_"
            f"{label}_"
            f"{count}.png"
        )

        crop_path = (
            Path(output_dir)
            / filename
        )

        save_crop(
            image,
            region["box_2d"],
            crop_path,
        )

        region["crop"] = str(
            crop_path
        )

        print(
            f"  Saved: {crop_path}"
        )

    # -----------------------------------------------------
    # Save visualization
    # -----------------------------------------------------

    visualization_path = (
        Path(output_dir)
        / f"{drawing_id}_localized.png"
    )

    save_visualization(
        image,
        regions,
        visualization_path,
    )

    print()
    print(
        f"Visualization: "
        f"{visualization_path}"
    )

    # -----------------------------------------------------
    # Save JSON
    # -----------------------------------------------------

    output_data = {
        "drawing_id": drawing_id,
        "regions": regions,
    }

    output_file = (
        Path(output_dir)
        / f"{drawing_id}_regions.json"
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
    print("Localization completed.")
    print("=" * 60)
    print(
        f"JSON          : {output_file}"
    )
    print(
        f"Visualization : "
        f"{visualization_path}"
    )
    print("=" * 60)

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
            "Localize regions in a manufacturing "
            "drawing using Florence-2."
        )
    )

    parser.add_argument(
        "pdf_path",
        help=(
            "Path to the input manufacturing "
            "drawing PDF."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Directory for localization outputs.",
    )

    args = parser.parse_args()

    localize_drawing(
        args.pdf_path,
        args.output_dir,
    )