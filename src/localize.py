"""
Stage 1 — Localize views and title block on a full-page drawing.

Usage:
    python src/localize.py --input reference_samples/ --out outputs/json/ --crops outputs/crops/

Requires OPENROUTER_API_KEY set in .env (see .env.example).
Converts each PDF in --input to a PNG, sends it + the grounding prompt to
Qwen2.5-VL via OpenRouter, parses the returned boxes (normalized 0-1000,
XYXY), converts them to pixel coords, saves a JSON file of boxes AND crops
each region out as its own PNG (feeding Stage 2 / Stage 3 later).
"""

import argparse
import base64
import json
import os
import time
from pathlib import Path

import fitz  # pymupdf
import requests
from dotenv import load_dotenv
from PIL import Image
import io

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_ID = "qwen/qwen2.5-vl-72b-instruct"  # paid: $0.10/M in, $0.40/M out
# For free testing (rate-limited): "qwen/qwen2.5-vl-72b-instruct:free"

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "localize_prompt.txt"


def pdf_to_png_bytes(pdf_path: Path, dpi: int = 300) -> bytes:
    doc = fitz.open(pdf_path)
    page = doc[0]
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes("png")


def call_qwen_grounding(image_bytes: bytes, prompt: str) -> tuple[list, float]:
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Copy .env.example to .env and fill it in."
        )

    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "model": MODEL_ID,
        "temperature": 0,
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                    },
                ],
            }
        ],
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    start = time.time()
    resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
    elapsed = time.time() - start
    resp.raise_for_status()

    data = resp.json()
    raw_text = data["choices"][0]["message"]["content"]

    cleaned = raw_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):]
    if cleaned.startswith("```"):
        cleaned = cleaned[len("```"):]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-len("```")]
    cleaned = cleaned.strip()

    try:
        boxes = json.loads(cleaned)
    except json.JSONDecodeError:
        boxes = [{"label": "PARSE_ERROR", "box_2d": [0, 0, 0, 0], "conf": 0.0, "raw": raw_text}]

    return boxes, round(elapsed, 2)


def normalized_to_pixels(box_2d, img_width, img_height):
    """Qwen boxes are XYXY normalized to 0-1000. Convert to pixel coords."""
    x0, y0, x1, y1 = box_2d
    px0 = int(x0 / 1000 * img_width)
    py0 = int(y0 / 1000 * img_height)
    px1 = int(x1 / 1000 * img_width)
    py1 = int(y1 / 1000 * img_height)
    return px0, py0, px1, py1


def call_qwen_grounding_with_retry(image_bytes: bytes, prompt: str, max_retries: int = 3) -> tuple[list, float]:
    """Wraps call_qwen_grounding with retry on 402 (in-flight budget) errors."""
    for attempt in range(1, max_retries + 1):
        try:
            return call_qwen_grounding(image_bytes, prompt)
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 402 and attempt < max_retries:
                wait_seconds = 15
                print(f"  402 in-flight budget hit, waiting {wait_seconds}s before retry "
                      f"(attempt {attempt}/{max_retries})...")
                time.sleep(wait_seconds)
                continue
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Folder of PDF drawings")
    parser.add_argument("--out", required=True, help="Folder to write box JSON")
    parser.add_argument("--crops", required=True, help="Folder to write cropped region PNGs")
    args = parser.parse_args()

    input_dir = Path(args.input)
    out_dir = Path(args.out)
    crops_dir = Path(args.crops)
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    prompt = PROMPT_PATH.read_text()

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {input_dir}. Place your reference drawings there first.")
        return

    for pdf_path in pdf_files:
        print(f"Localizing {pdf_path.name} ...")
        image_bytes = pdf_to_png_bytes(pdf_path)
        pil_image = Image.open(io.BytesIO(image_bytes))
        img_w, img_h = pil_image.size

        boxes, elapsed = call_qwen_grounding_with_retry(image_bytes, prompt)

        drawing_id = pdf_path.stem
        result = {
            "drawing_id": drawing_id,
            "image_width": img_w,
            "image_height": img_h,
            "regions": [],
            "_latency_seconds": elapsed,
            "_model": MODEL_ID,
        }

        for i, box in enumerate(boxes):
            label = box.get("label", "unknown")
            conf = box.get("conf", None)
            box_2d = box.get("box_2d")

            region_entry = {"label": label, "conf": conf, "box_2d_normalized": box_2d}

            if box_2d and len(box_2d) == 4:
                px0, py0, px1, py1 = normalized_to_pixels(box_2d, img_w, img_h)
                region_entry["box_2d_pixels"] = [px0, py0, px1, py1]

                # Crop and save
                if px1 > px0 and py1 > py0:
                    crop = pil_image.crop((px0, py0, px1, py1))
                    crop_filename = f"{drawing_id}_{label}_{i}.png"
                    crop.save(crops_dir / crop_filename)
                    region_entry["crop_file"] = crop_filename

            result["regions"].append(region_entry)

        out_path = out_dir / f"{drawing_id}_regions.json"
        out_path.write_text(json.dumps(result, indent=2))

        labels_found = [r["label"] for r in result["regions"]]
        print(f"  -> found {len(labels_found)} regions: {labels_found} ({elapsed}s)")
        print(f"  -> saved to {out_path}")
        time.sleep(3)  # small pause between drawings to avoid stacking in-flight requests


if __name__ == "__main__":
    main()
