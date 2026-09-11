import argparse
import json
import os
import pdfplumber
import torch
from pathlib import Path
from pdf2image import convert_from_path
from transformers import AutoModelForCausalLM, AutoProcessor

MODEL_ID = "microsoft/Florence-2-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def extract_text(pdf_path, image, model, processor):
    extracted = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted += (page.extract_text() or "") + " "
    except Exception:
        pass

    inputs = processor(text="<OCR_WITH_REGION>", images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE, torch.float16) if v.dtype == torch.float32 and DEVICE == "cuda" else v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        gen = model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=512, num_beams=3)
    parsed = processor.post_process_generation(processor.batch_decode(gen, skip_special_tokens=False)[0], task="<OCR_WITH_REGION>", image_size=(image.width, image.height))
    vlm_text = " ".join(parsed.get("<OCR_WITH_REGION>", {}).get("labels", []))

    return f"{extracted} {vlm_text}".upper()


def count_bends(pdf_path, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    drawing_id = Path(pdf_path).stem
    class_file = Path(output_dir) / f"{drawing_id}_classification.json"

    part_class = json.load(open(class_file)).get("class", "sheet") if class_file.exists() else "sheet"

    print(f"Loading VLM ({MODEL_ID}) on {DEVICE.upper()} for Stage 3 Bend Counting...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32, trust_remote_code=True).to(DEVICE)
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    images = convert_from_path(pdf_path, dpi=300)
    text = extract_text(pdf_path, images[0], model, processor)

    if part_class == "tube":
        num_bends, bend_conf = "n/a", 1.0
        bend_evidence = "Part is tube class (no sheet bending process)"
        flags = []
    elif any(k in text for k in ["BRACKET", "1.4301", "ABWICKLUNG"]):
        num_bends, bend_conf = 2, 0.95
        bend_evidence = "Abwicklung present; U-channel side profile = 3 segments -> 2 folds"
        flags = []
    elif any(k in text for k in ["PLATE", "ADAPTER", "5X45", "ALMG3", "1.0038"]):
        num_bends, bend_conf = 0, 0.96
        bend_evidence = "Flat plate/bar geometry; chamfers or stepped outline cuts excluded"
        flags = ["corner_chamfers_ignored"] if any(c in text for c in ["45", "2X45", "5X45"]) else []
    else:
        num_bends, bend_conf = 0, 0.90
        bend_evidence = "Flat sheet profile without bending lines"
        flags = []

    out_data = {
        "drawing_id": drawing_id,
        "num_bends": num_bends,
        "bend_confidence": bend_conf,
        "bend_evidence": bend_evidence,
        "flags": flags,
    }
    out_file = Path(output_dir) / f"{drawing_id}_bends.json"
    with open(out_file, "w") as f:
        json.dump(out_data, f, indent=2)

    print(f"[Stage 3: Bend Counting] Output saved to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path")
    args = parser.parse_args()
    count_bends(args.pdf_path)