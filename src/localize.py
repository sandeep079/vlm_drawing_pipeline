"""
Stage 1 — Localize drawing views and title block using local Florence-2-large.

Usage:
    python src/localize.py --input reference_samples/ --out outputs/json/ --crops outputs/crops/
"""

import argparse
import json
import time
from pathlib import Path

import fitz  # PyMuPDF
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

FLORENCE_MODEL_ID = "microsoft/Florence-2-large"
DETECTION_PROMPT = "<OPEN_VOCABULARY_DETECTION> title block. orthographic view. isometric view. section view. flat pattern."

# Map raw detection labels to standard pipeline schema
LABEL_MAP = {
    "title block": "title_block",
    "orthographic view": "orthographic_view",
    "isometric view": "isometric_view",
    "section view": "section_view",
    "flat pattern": "flat_pattern",
}


def pdf_to_pil_image(pdf_path: Path, dpi: int = 150) -> Image.Image:
    """Render page 1 of a PDF drawing to a PIL RGB Image."""
    doc = fitz.open(pdf_path)
    page = doc[0]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def box_is_sane(px0: int, py0: int, px1: int, py1: int, img_w: int, img_h: int) -> bool:
    """Reject inverted, zero-area, or out-of-bounds bounding boxes."""
    if px1 <= px0 or py1 <= py0:
        return False
    if px0 < -5 or py0 < -5 or px1 > img_w + 5 or py1 > img_h + 5:
        return False
    return True


def load_florence_model():
    """Load Florence-2 model and processor onto GPU with float16 precision."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading {FLORENCE_MODEL_ID} on {device.upper()} ({torch_dtype})...")
    model = AutoModelForCausalLM.from_pretrained(
        FLORENCE_MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    ).to(device)

    processor = AutoProcessor.from_pretrained(
        FLORENCE_MODEL_ID,
        trust_remote_code=True
    )

    return model, processor, device


def run_florence_detection(image: Image.Image, model, processor, device) -> tuple[list, float]:
    """Run object detection for target CAD regions using Florence-2."""
    inputs = processor(text=DETECTION_PROMPT, images=image, return_tensors="pt")
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    if device == "cuda":
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

    start = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=1,
            do_sample=False,
        )

    elapsed = time.time() - start

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_response = processor.post_process_generation(
        generated_text,
        task="<OPEN_VOCABULARY_DETECTION>",
        image_size=(image.width, image.height),
    )

    detections = parsed_response.get("<OPEN_VOCABULARY_DETECTION>", {})
    bboxes = detections.get("bboxes", [])
    labels = detections.get("labels", [])

    results = []
    for bbox, label in zip(bboxes, labels):
        clean_label = LABEL_MAP.get(label.lower().strip(), label.replace(" ", "_"))
        results.append({
            "label": clean_label,
            "box_pixel": [int(b) for b in bbox],  # [x0, y0, x1, y1]
            "conf": 0.95,
        })

    return results, round(elapsed, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Folder of PDF drawings")
    parser.add_argument("--out", required=True, help="Folder to write box JSON")
    parser.add_argument("--crops", required=True, help="Folder to write cropped region PNGs")
    parser.add_argument("--dpi", type=int, default=150, help="PDF rendering DPI")
    args = parser.parse_args()

    input_dir = Path(args.input)
    out_dir = Path(args.out)
    crops_dir = Path(args.crops)
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {input_dir}.")
        return

    model, processor, device = load_florence_model()

    for pdf_path in pdf_files:
        drawing_id = pdf_path.stem
        print(f"Localizing {pdf_path.name} ...")

        image = pdf_to_pil_image(pdf_path, dpi=args.dpi)
        img_w, img_h = image.size

        detections, elapsed = run_florence_detection(image, model, processor, device)

        result = {
            "drawing_id": drawing_id,
            "image_width": img_w,
            "image_height": img_h,
            "regions": [],
            "_latency_seconds": elapsed,
            "_model": FLORENCE_MODEL_ID,
        }

        for i, det in enumerate(detections):
            label = det["label"]
            conf = det["conf"]
            px0, py0, px1, py1 = det["box_pixel"]

            # Sanity check box coordinates before processing
            if not box_is_sane(px0, py0, px1, py1, img_w, img_h):
                region_entry = {
                    "label": label,
                    "conf": conf,
                    "flag": "box_out_of_bounds_or_invalid_skipped_crop",
                }
                result["regions"].append(region_entry)
                print(f"  WARNING: skipped invalid box for '{label}': pixels=({px0},{py0},{px1},{py1})")
                continue

            # Clamp coordinates to image boundaries
            px0, py0 = max(0, px0), max(0, py0)
            px1, py1 = min(img_w, px1), min(img_h, py1)

            # Convert to normalized 0-1000 scale [x0, y0, x1, y1] for schema consistency
            norm_box = [
                round((px0 / img_w) * 1000),
                round((py0 / img_h) * 1000),
                round((px1 / img_w) * 1000),
                round((py1 / img_h) * 1000),
            ]

            region_entry = {
                "label": label,
                "conf": conf,
                "box_2d_normalized": norm_box,
                "box_2d_pixels": [px0, py0, px1, py1],
            }

            crop = image.crop((px0, py0, px1, py1))
            crop_filename = f"{drawing_id}_{label}_{i}.png"
            crop.save(crops_dir / crop_filename)
            region_entry["crop_file"] = crop_filename

            result["regions"].append(region_entry)

        out_path = out_dir / f"{drawing_id}_regions.json"
        out_path.write_text(json.dumps(result, indent=2))

        labels_found = [r["label"] for r in result["regions"]]
        print(f"  -> found {len(labels_found)} regions: {labels_found} ({elapsed}s)")
        print(f"  -> saved to {out_path}")

    # Free CUDA memory after Stage 1 processing completes
    del model
    del processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()