import os
import sys
import json
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForCausalLM
from pdf2image import convert_from_path

MODEL_ID = "microsoft/Florence-2-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Separate queries yield tighter grounding boxes than chained prompts
TARGET_PROMPTS = [
    "<CAPTION_TO_PHRASE_GROUNDING> title block",
    "<CAPTION_TO_PHRASE_GROUNDING> flat pattern",
    "<CAPTION_TO_PHRASE_GROUNDING> isometric view",
    "<CAPTION_TO_PHRASE_GROUNDING> orthographic view"
]

print("Loading Florence-2 model onto GPU...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16, 
    trust_remote_code=True,
    attn_implementation="sdpa"
).to(DEVICE)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)


def run_pipeline(image_path, min_box_area=2000, max_coverage_ratio=0.85, output_dir="crops", json_out="drawing_analysis.json"):
    if not os.path.exists(image_path):
        print(f"Error: '{image_path}' not found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    page_area = image.width * image.height
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)

    all_detections = []

    # --- Stage 1: Targeted Phrase Grounding Iteration ---
    for task_prompt in TARGET_PROMPTS:
        inputs = processor(text=task_prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(DEVICE, torch.float16) if v.dtype == torch.float32 else v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            gen_out = model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=1024, num_beams=3)
        
        parsed = processor.post_process_generation(
            processor.batch_decode(gen_out, skip_special_tokens=False)[0],
            task="<CAPTION_TO_PHRASE_GROUNDING>",
            image_size=(image.width, image.height)
        ).get("<CAPTION_TO_PHRASE_GROUNDING>", {})

        for bbox, label in zip(parsed.get("bboxes", []), parsed.get("labels", [])):
            all_detections.append((bbox, label))

    pipeline_results = []
    retained_count = 0

    # --- Stage 2: Spatial Filtering, Cropping & OCR ---
    print(f"[Stage 2] Filtering {len(all_detections)} candidate detections...")
    for idx, (bbox, label) in enumerate(all_detections):
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        coverage_ratio = area / page_area

        # Skip noise boxes OR full-page canvas bounding boxes
        if area < min_box_area:
            print(f"  -> Skipped '{label}': below min area ({int(area)}px)")
            continue
        if coverage_ratio > max_coverage_ratio:
            print(f"  -> Skipped '{label}': spans full sheet ({coverage_ratio:.1%} of page)")
            continue

        retained_count += 1
        clean_label = label.strip().replace(" ", "_")

        draw.rectangle(bbox, outline="red", width=3)
        draw.text((x1, max(0, y1 - 10)), label, fill="red")

        crop = image.crop(bbox)
        crop_path = os.path.join(output_dir, f"crop_{idx}_{clean_label}.png")
        crop.save(crop_path)

        # Stage 2 OCR
        inputs_ocr = processor(text="<OCR_WITH_REGION>", images=crop, return_tensors="pt")
        inputs_ocr = {k: v.to(DEVICE, torch.float16) if v.dtype == torch.float32 else v.to(DEVICE) for k, v in inputs_ocr.items()}

        with torch.no_grad():
            gen_ocr = model.generate(input_ids=inputs_ocr["input_ids"], pixel_values=inputs_ocr["pixel_values"], max_new_tokens=1024, num_beams=3)

        parsed_ocr = processor.post_process_generation(
            processor.batch_decode(gen_ocr, skip_special_tokens=False)[0],
            task="<OCR_WITH_REGION>",
            image_size=(crop.width, crop.height)
        ).get("<OCR_WITH_REGION>", {})

        pipeline_results.append({
            "crop_id": idx,
            "label": label,
            "bbox": bbox,
            "page_coverage_pct": round(coverage_ratio * 100, 2),
            "crop_file": crop_path,
            "ocr_text": parsed_ocr.get("labels", [])
        })

    annotated_image.save(f"annotated_{os.path.basename(image_path)}")
    with open(json_out, "w") as f:
        json.dump(pipeline_results, f, indent=4)

    print(f"\nExecution Summary:")
    print(f"--> Retained {retained_count} bounded CAD regions (Full-page canvas boxes dropped).")
    print(f"--> Exported JSON: {json_out}")


def process_pdf(pdf_path):
    images = convert_from_path(pdf_path, dpi=300)
    for i, img in enumerate(images):
        filename = f"page_{i+1}.png"
        img.save(filename, "PNG")
        run_pipeline(filename, output_dir=f"crops_page_{i+1}", json_out=f"analysis_page_{i+1}.json")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "reference_samples/001_redacted.pdf"
    if target.endswith(".pdf"):
        process_pdf(target)
    else:
        run_pipeline(target)