"""
Stage 1 — Localize views and title block on a full-page drawing.

Usage:
    python src/localize.py --input reference_samples/ --out outputs/json/ --crops outputs/crops/
"""

import argparse
import base64
import io
import json
import time
from pathlib import Path

import fitz  # PyMuPDF
import requests
from PIL import Image

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5vl:7b"
PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "localize_prompt.txt"


def pdf_to_png_bytes(pdf_path: Path, dpi: int = 100) -> bytes:
    """Renders the first page of a PDF to PNG image bytes at specified DPI."""
    doc = fitz.open(pdf_path)
    page = doc[0]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes("png")


def normalized_to_pixels(box_2d: list, img_width: int, img_height: int, pad_px: int = 15) -> tuple[int, int, int, int]:
    """
    Converts 0-1000 normalized coordinates [ymin, xmin, ymax, xmax]
    to absolute pixel bounds [x0, y0, x1, y1] with margin padding.
    """
    ymin, xmin, ymax, xmax = box_2d

    px0 = max(0, int(xmin / 1000.0 * img_width) - pad_px)
    py0 = max(0, int(ymin / 1000.0 * img_height) - pad_px)
    px1 = min(img_width, int(xmax / 1000.0 * img_width) + pad_px)
    py1 = min(img_height, int(ymax / 1000.0 * img_height) + pad_px)

    return px0, py0, px1, py1


def call_ollama_grounding(image_bytes: bytes, prompt: str, model_name: str = DEFAULT_MODEL) -> tuple[list, float]:
    """Sends image and grounding prompt to local Ollama chat endpoint with extended timeout."""
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [b64_image],
            }
        ],
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.0,
        },
    }

    start = time.time()
    try:
        # Timeout set to 300s (5 mins) to handle cold model loading and GPU vision encoding
        resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Could not connect to Ollama server at {OLLAMA_URL}. "
            f"Ensure Ollama server is running in the background."
        )
    except requests.exceptions.ReadTimeout:
        raise RuntimeError(
            f"Ollama server timed out after 300s processing '{model_name}'. "
            f"The image or hardware execution took too long."
        )

    elapsed = time.time() - start
    data = resp.json()
    raw_text = data.get("message", {}).get("content", "").strip()

    cleaned = raw_text
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):]
    if cleaned.startswith("```"):
        cleaned = cleaned[len("```"):]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-len("```")]
    cleaned = cleaned.strip()

    try:
        boxes = json.loads(cleaned)
        if isinstance(boxes, dict):
            boxes = boxes.get("regions", boxes.get("boxes", [boxes]))
    except json.JSONDecodeError:
        boxes = [{"label": "PARSE_ERROR", "box_2d": [0, 0, 0, 0], "conf": 0.0, "raw": raw_text}]

    return boxes, round(elapsed, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Folder of PDF drawings")
    parser.add_argument("--out", required=True, help="Folder to write box JSON")
    parser.add_argument("--crops", required=True, help="Folder to write cropped region PNGs")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--dpi", type=int, default=100, help="PDF rendering DPI")
    args = parser.parse_args()

    input_dir = Path(args.input)
    out_dir = Path(args.out)
    crops_dir = Path(args.crops)
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"Prompt file not found at {PROMPT_PATH}")

    prompt = PROMPT_PATH.read_text()

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {input_dir}. Place your reference drawings there first.")
        return

    for pdf_path in pdf_files:
        print(f"Localizing {pdf_path.name} using model '{args.model}'...")
        image_bytes = pdf_to_png_bytes(pdf_path, dpi=args.dpi)
        pil_image = Image.open(io.BytesIO(image_bytes))
        img_w, img_h = pil_image.size

        boxes, elapsed = call_ollama_grounding(image_bytes, prompt, model_name=args.model)

        drawing_id = pdf_path.stem
        result = {
            "drawing_id": drawing_id,
            "image_width": img_w,
            "image_height": img_h,
            "regions": [],
            "_latency_seconds": elapsed,
            "_model": args.model,
        }

        for i, box in enumerate(boxes):
            label = box.get("label", "unknown")
            conf = box.get("conf", 0.9)
            box_2d = box.get("box_2d")

            region_entry = {"label": label, "conf": conf, "box_2d_normalized": box_2d}

            if box_2d and len(box_2d) == 4:
                px0, py0, px1, py1 = normalized_to_pixels(box_2d, img_w, img_h, pad_px=15)
                region_entry["box_2d_pixels"] = [px0, py0, px1, py1]

                if px1 > px0 and py1 > py0:
                    crop = pil_image.crop((px0, py0, px1, py1))
                    crop_filename = f"{drawing_id}_{label}_{i}.png"
                    crop.save(crops_dir / crop_filename)
                    region_entry["crop_file"] = crop_filename

            result["regions"].append(region_entry)

        out_path = out_dir / f"{drawing_id}_regions.json"
        out_path.write_text(json.dumps(result, indent=2))

        labels_found = [r["label"] for r in result["regions"]]
        print(f"  -> Found {len(labels_found)} regions in {elapsed}s: {labels_found}")
        print(f"  -> Saved output to {out_path}")


if __name__ == "__main__":
    main()