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

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_ID = "qwen/qwen2.5-vl-72b-instruct"  # paid: $0.10/M in, $0.40/M out

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "count_bends_prompt.txt"

RELEVANT_LABELS = ["flat_pattern", "isometric_view", "orthographic_view", "section_view"]


def pdf_to_png_bytes(pdf_path: Path, dpi: int = 300) -> bytes:
    doc = fitz.open(pdf_path)
    page = doc[0]
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes("png")


def is_crop_usable(crop_path: Path) -> bool:
    """Reject 0-byte files and near-solid-black images (broken/out-of-bounds crops)."""
    if not crop_path.exists() or crop_path.stat().st_size == 0:
        return False
    try:
        img = Image.open(crop_path).convert("L")  # grayscale
        extrema = img.getextrema()
        # Solid black: min and max pixel value both near 0
        if extrema[1] < 10:
            return False
    except Exception:
        return False
    return True


def find_usable_crops(drawing_id: str, crops_dir: Path) -> list[Path]:
    found = []
    for label in RELEVANT_LABELS:
        matches = sorted(crops_dir.glob(f"{drawing_id}_{label}_*.png"))
        for m in matches:
            if is_crop_usable(m):
                found.append(m)
                break  # one usable crop per label type is enough
    return found


def image_to_b64(path_or_bytes) -> str:
    if isinstance(path_or_bytes, bytes):
        return base64.b64encode(path_or_bytes).decode("utf-8")
    return base64.b64encode(Path(path_or_bytes).read_bytes()).decode("utf-8")


def call_qwen_count_bends(image_b64_list: list[str], prompt: str, max_retries: int = 4) -> tuple[dict, float]:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set. Copy .env.example to .env and fill it in.")

    content = [{"type": "text", "text": prompt}]
    for b64 in image_b64_list:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

    payload = {
        "model": MODEL_ID,
        "temperature": 0,
        "max_tokens": 512,
        "messages": [{"role": "user", "content": content}],
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    start = time.time()
    for attempt in range(1, max_retries + 1):
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
        if resp.status_code == 402 and attempt < max_retries:
            wait_seconds = 20 * attempt  # 20s, 40s, 60s backoff
            print(f"  402 in-flight budget hit, waiting {wait_seconds}s before retry "
                  f"(attempt {attempt}/{max_retries})...")
            time.sleep(wait_seconds)
            continue
        break
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
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        result = {"num_bends": None, "bend_confidence": 0.0,
                   "bend_evidence": f"PARSE_ERROR: {raw_text}", "flags": ["json_parse_error"]}

    return result, round(elapsed, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Folder of PDF drawings")
    parser.add_argument("--crops", required=True, help="Folder of Stage 1 crop PNGs")
    parser.add_argument("--out", required=True, help="Folder to write bend-count JSON")
    args = parser.parse_args()

    input_dir = Path(args.input)
    crops_dir = Path(args.crops)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt = PROMPT_PATH.read_text()

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {input_dir}.")
        return

    for pdf_path in pdf_files:
        drawing_id = pdf_path.stem
        print(f"Counting bends for {drawing_id} ...")

        usable_crops = find_usable_crops(drawing_id, crops_dir)
        flags = []

        if usable_crops:
            print(f"  using {len(usable_crops)} Stage-1 crop(s): {[c.name for c in usable_crops]}")
            image_b64_list = [image_to_b64(c) for c in usable_crops]
        else:
            print("  no usable Stage-1 crops found -> falling back to full page")
            flags.append("fallback_to_full_page_stage1_crops_missing_or_broken")
            full_page_bytes = pdf_to_png_bytes(pdf_path)
            image_b64_list = [image_to_b64(full_page_bytes)]

        result, elapsed = call_qwen_count_bends(image_b64_list, prompt)
        result["drawing_id"] = drawing_id
        result["_latency_seconds"] = elapsed
        result["_model"] = MODEL_ID
        result.setdefault("flags", [])
        result["flags"].extend(flags)

        out_path = out_dir / f"{drawing_id}_bends.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"  -> num_bends={result.get('num_bends')} "
              f"(conf={result.get('bend_confidence')}, {elapsed}s) saved to {out_path}")

        time.sleep(3)


if __name__ == "__main__":
    main()
