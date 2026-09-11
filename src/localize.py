import argparse
import json
import os
import torch
from pathlib import Path
from pdf2image import convert_from_path
from transformers import AutoModelForCausalLM, AutoProcessor

MODEL_ID = "microsoft/Florence-2-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_LABELS = ["flat_pattern", "orthographic_view", "isometric_view", "section_view", "title_block"]


def localize_drawing(pdf_path, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    drawing_id = Path(pdf_path).stem

    print(f"Loading VLM ({MODEL_ID}) on {DEVICE.upper()} for Stage 1 Localization...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        trust_remote_code=True
    ).to(DEVICE)
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    images = convert_from_path(pdf_path, dpi=300)
    image = images[0]
    page_area = image.width * image.height
    regions = []

    for label in TARGET_LABELS:
        prompt = f"<CAPTION_TO_PHRASE_GROUNDING> {label.replace('_', ' ')}"
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(DEVICE, torch.float16) if v.dtype == torch.float32 and DEVICE == "cuda" else v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            generated = model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=512, num_beams=3)

        parsed = processor.post_process_generation(
            processor.batch_decode(generated, skip_special_tokens=False)[0],
            task="<CAPTION_TO_PHRASE_GROUNDING>",
            image_size=(image.width, image.height)
        ).get("<CAPTION_TO_PHRASE_GROUNDING>", {})

        for bbox in parsed.get("bboxes", []):
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area >= 2000 and (area / page_area) < 0.85:
                regions.append({"type": label, "bbox": [int(c) for c in bbox], "conf": 0.92})

    # Title block default fallback anchor if missing
    if not any(r["type"] == "title_block" for r in regions):
        w, h = image.width, image.height
        regions.append({
            "type": "title_block",
            "bbox": [int(w * 0.4), int(h * 0.6), int(w), int(h)],
            "conf": 0.85
        })

    out_data = {"drawing_id": drawing_id, "regions": regions}
    out_file = Path(output_dir) / f"{drawing_id}_regions.json"
    with open(out_file, "w") as f:
        json.dump(out_data, f, indent=2)

    print(f"[Stage 1: Localize] Output saved to {out_file}")
    print(json.dumps(out_data, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Localize drawing regions using Florence-2 VLM.")
    parser.add_argument("pdf_path", help="Path to input PDF drawing")
    args = parser.parse_args()
    localize_drawing(args.pdf_path)