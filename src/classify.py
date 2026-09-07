"""
Stage 2 — Sheet vs. Tube classification.

Usage:
    python src/classify.py --input reference_samples/ --out outputs/json/

Requires OPENROUTER_API_KEY set in .env (see .env.example).
Converts each PDF in --input to a PNG, sends it + the classification prompt
to Qwen2.5-VL via OpenRouter, and writes one JSON result per drawing.
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

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_ID = "qwen/qwen2.5-vl-72b-instruct"  # $0.10/M in, $0.40/M out on OpenRouter
# Free tier for testing (rate-limited, may be lower quality): "qwen/qwen2.5-vl-72b-instruct:free"

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "classify_prompt.txt"


def pdf_to_png_bytes(pdf_path: Path, dpi: int = 300) -> bytes:
    """Render page 1 of a single-part PDF drawing to PNG bytes."""
    doc = fitz.open(pdf_path)
    page = doc[0]
    zoom = dpi / 72  # PDF base is 72 DPI
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes("png")


def classify_drawing(image_bytes: bytes, prompt: str) -> dict:
    """Send one drawing image + prompt to Qwen2.5-VL via OpenRouter, return parsed JSON."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Copy .env.example to .env and fill it in."
        )

    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "model": MODEL_ID,
        "temperature": 0,
        "max_tokens": 512,
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

    # Strip accidental markdown fences before parsing
    cleaned = raw_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):]
    if cleaned.startswith("```"):
        cleaned = cleaned[len("```"):]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-len("```")]
    cleaned = cleaned.strip()

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        result = {"class": None, "class_confidence": 0.0, "class_evidence": f"PARSE_ERROR: {raw_text}"}

    result["_latency_seconds"] = round(elapsed, 2)
    result["_model"] = MODEL_ID
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Folder of PDF drawings")
    parser.add_argument("--out", required=True, help="Folder to write JSON results")
    args = parser.parse_args()

    input_dir = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt = PROMPT_PATH.read_text()

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {input_dir}. Place your reference drawings there first.")
        return

    for pdf_path in pdf_files:
        print(f"Classifying {pdf_path.name} ...")
        image_bytes = pdf_to_png_bytes(pdf_path)
        result = classify_drawing(image_bytes, prompt)
        result["drawing_id"] = pdf_path.stem

        time.sleep(10)
        
        out_path = out_dir / f"{pdf_path.stem}_classification.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"  -> {result.get('class')} (conf={result.get('class_confidence')}, "
              f"{result.get('_latency_seconds')}s) saved to {out_path}")


if __name__ == "__main__":
    main()
