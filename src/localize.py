"""
Stage 1 — Localize views and title block using Florence-2-large (Local GPU).

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


def pdf_to_pil_image(pdf_path: Path, dpi: int = 150) -> Image.Image:
    """Renders the first page of a PDF to a PIL Image at specified DPI."""
    doc = fitz.open(pdf_path)
    page = doc[0]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def load_florence_model():
    """Loads Florence-2 model in fp16 precision on CUDA."""
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
    """Runs open-vocabulary detection for CAD drawing components."""
    prompt = "<OPEN_VOCABULARY_DETECTION> title block. orthographic view. isometric view. section view. flat pattern."

    inputs = processor(text=prompt, images=image, return_tensors="pt")
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
    # Map raw detection labels to target schema labels
    label_map = {
        "title block": "title_block",
        "orthographic view": "orthographic_view",
        "isometric view": "isometric_view",
        "section view": "section_view",
        "flat pattern": "flat_pattern",
    }

    for bbox, label in zip(bboxes, labels):
        clean_label = label_map.get(label.lower().strip(), label.replace(" ", "_"))
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

    input_dir, out_dir, crops_dir = Path(args.input), Path(args.out), Path(args.crops)
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {input_dir}.")
        return

    model, processor, device = load_florence_model()

    for pdf_path in pdf_files:
        print(f"Localizing {pdf_path.name}...")
        image = pdf_to_pil_image(pdf_path, dpi=args.dpi)
        img_w, img_h = image.size

        detections, elapsed = run_florence_detection(image, model, processor, device)

        drawing_id = pdf_path.stem
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
            x0, y0, x1, y1 = det["box_pixel"]

            # Clamp coordinates to image boundaries
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(img_w, x1), min(img_h, y1)

            # Convert to normalized 0-1000 [ymin, xmin, ymax, xmax]
            norm_box = [
                round((y0 / img_h) * 1000),
                round((x0 / img_w) * 1000),
                round((y1 / img_h) * 1000),
                round((x1 / img_w) * 1000),
            ]

            region_entry = {
                "label": label,
                "conf": det["conf"],
                "box_2d_normalized": norm_box,
                "box_2d_pixels": [x0, y0, x1, y1],
            }

            if x1 > x0 and y1 > y0:
                crop = image.crop((x0, y0, x1, y1))
                crop_filename = f"{drawing_id}_{label}_{i}.png"
                crop.save(crops_dir / crop_filename)
                region_entry["crop_file"] = crop_filename

            result["regions"].append(region_entry)

        out_path = out_dir / f"{drawing_id}_regions.json"
        out_path.write_text(json.dumps(result, indent=2))

        labels_found = [r["label"] for r in result["regions"]]
        print(f"  -> Found {len(labels_found)} regions in {elapsed}s: {labels_found}")

    # Free memory after Stage 1 finishes
    del model
    del processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()