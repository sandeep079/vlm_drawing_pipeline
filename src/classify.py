"""
Stage 2 — Sheet vs. Tube classification using local Ollama (qwen2-vl:2b).

Usage:
    python src/classify.py --input reference_samples/ --out outputs/json/
"""

import argparse
import base64
import json
import time
from pathlib import Path

import fitz  # PyMuPDF
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5vl:3b"
PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "classify_prompt.txt"


def pdf_to_png_bytes(pdf_path: Path, dpi: int = 200) -> bytes:
    """Render page 1 of a PDF drawing to PNG bytes."""
    doc = fitz.open(pdf_path)
    page = doc[0]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes("png")


def classify_drawing(image_bytes: bytes, prompt: str, model_name: str = DEFAULT_MODEL) -> dict:
    """Send image and classification prompt directly to local Ollama chat endpoint."""
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
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Could not connect to Ollama at {OLLAMA_URL}. Ensure Ollama is running (`ollama serve`)."
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
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        result = {
            "class": "unknown",
            "class_confidence": 0.0,
            "class_evidence": f"PARSE_ERROR: {raw_text}",
        }

    result["_latency_seconds"] = round(elapsed, 2)
    result["_model"] = model_name
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Folder of PDF drawings")
    parser.add_argument("--out", required=True, help="Folder to write JSON results")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--dpi", type=int, default=200, help="PDF rendering DPI")
    args = parser.parse_args()

    input_dir = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"Prompt file missing at {PROMPT_PATH}")

    prompt = PROMPT_PATH.read_text()

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {input_dir}.")
        return

    for pdf_path in pdf_files:
        print(f"Classifying {pdf_path.name} using '{args.model}'...")
        image_bytes = pdf_to_png_bytes(pdf_path, dpi=args.dpi)
        
        result = classify_drawing(image_bytes, prompt, model_name=args.model)
        result["drawing_id"] = pdf_path.stem

        out_path = out_dir / f"{pdf_path.stem}_classification.json"
        out_path.write_text(json.dumps(result, indent=2))

        print(
            f"  -> Class: {result.get('class')} "
            f"(conf={result.get('class_confidence')}, {result.get('_latency_seconds')}s)"
        )


if __name__ == "__main__":
    main()