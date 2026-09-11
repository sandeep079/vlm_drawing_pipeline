"""
Stage 3 — Count bends for sheet parts using local Ollama (qwen2.5vl:3b).

Usage:
    python src/count_bends.py --input reference_samples/ --crops outputs/crops/ --out outputs/json/
"""

import argparse
import base64
import json
import time
from pathlib import Path

import fitz  # PyMuPDF
import requests
from PIL import Image

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5vl:3b"
PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "count_bends_prompt.txt"
RELEVANT_LABELS = ["flat_pattern", "isometric_view", "orthographic_view", "section_view"]


def pdf_to_png_bytes(pdf_path: Path, dpi: int = 150) -> bytes:
    """Render page 1 of a PDF drawing to PNG bytes."""
    doc = fitz.open(pdf_path)
    page = doc[0]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes("png")


def is_crop_usable(crop_path: Path) -> bool:
    """Reject empty files or near-black cropped images."""
    if not crop_path.exists() or crop_path.stat().st_size == 0:
        return False
    try:
        img = Image.open(crop_path).convert("L")
        extrema = img.getextrema()
        if extrema[1] < 10:
            return False
    except Exception:
        return False
    return True


def find_usable_crops(drawing_id: str, crops_dir: Path) -> list[Path]:
    """Find valid Stage 1 crop files matching target view labels."""
    found = []
    for label in RELEVANT_LABELS:
        matches = sorted(crops_dir.glob(f"{drawing_id}_{label}_*.png"))
        for m in matches:
            if is_crop_usable(m):
                found.append(m)
                break
    return found


def call_ollama_count_bends(prompt: str, image_bytes_list: list[bytes], model_name: str = DEFAULT_MODEL) -> tuple[dict, float]:
    """Send image list + prompt to local Ollama chat endpoint."""
    b64_images = [base64.b64encode(b).decode("utf-8") for b in image_bytes_list]

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": b64_images,
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
        resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
        
        if resp.status_code == 404:
            raise RuntimeError(
                f"Model '{model_name}' not found in Ollama.\n"
                f"Run `ollama pull {model_name}` in your terminal first."
            )
        resp.raise_for_status()

    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Could not connect to Ollama at {OLLAMA_URL}. Ensure Ollama server is running (`ollama serve`)."
        )

    elapsed = time.time() - start
    data = resp.json()
    raw_text = data.get("message", {}).get("content", "").strip()

    # Clean markdown fences
    cleaned = raw_text
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = {
            "num_bends": None,
            "bend_confidence": 0.0,
            "bend_evidence": f"PARSE_ERROR: {raw_text}",
            "flags": ["json_parse_error"],
        }

    return parsed, round(elapsed, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Folder of PDF drawings")
    parser.add_argument("--crops", required=True, help="Folder of Stage 1 crop PNGs")
    parser.add_argument("--out", required=True, help="Folder to write bend-count JSON")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--dpi", type=int, default=150, help="PDF rendering DPI for fallback")
    args = parser.parse_args()

    input_dir = Path(args.input)
    crops_dir = Path(args.crops)
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
        drawing_id = pdf_path.stem
        
        # Skip bend counting if Stage 2 marked this part as tube
        class_file = out_dir / f"{drawing_id}_classification.json"
        if class_file.exists():
            try:
                class_data = json.loads(class_file.read_text())
                if class_data.get("class") == "tube":
                    print(f"Skipping bend count for {drawing_id} (classified as tube).")
                    result = {
                        "drawing_id": drawing_id,
                        "num_bends": "n/a",
                        "bend_confidence": 1.0,
                        "bend_evidence": "Part classified as tube in Stage 2.",
                        "flags": ["skipped_tube_part"],
                        "_latency_seconds": 0.0,
                        "_model": args.model,
                    }
                    out_path = out_dir / f"{drawing_id}_bends.json"
                    out_path.write_text(json.dumps(result, indent=2))
                    continue
            except Exception:
                pass

        print(f"Counting bends for {drawing_id} using '{args.model}'...")

        usable_crops = find_usable_crops(drawing_id, crops_dir)
        flags = []

        if usable_crops:
            print(f"  using {len(usable_crops)} Stage-1 crop(s): {[c.name for c in usable_crops]}")
            image_bytes_list = [c.read_bytes() for c in usable_crops]
        else:
            print("  no usable Stage-1 crops found -> falling back to full page")
            flags.append("fallback_to_full_page_stage1_crops_missing_or_broken")
            image_bytes_list = [pdf_to_png_bytes(pdf_path, dpi=args.dpi)]

        result, elapsed = call_ollama_count_bends(prompt, image_bytes_list, model_name=args.model)
        
        result["drawing_id"] = drawing_id
        result["_latency_seconds"] = elapsed
        result["_model"] = args.model
        result.setdefault("flags", [])
        result["flags"].extend(flags)

        out_path = out_dir / f"{drawing_id}_bends.json"
        out_path.write_text(json.dumps(result, indent=2))
        
        print(
            f"  -> num_bends={result.get('num_bends')} "
            f"(conf={result.get('bend_confidence')}, {elapsed}s) saved to {out_path}"
        )


if __name__ == "__main__":
    main()