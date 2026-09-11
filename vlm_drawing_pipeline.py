import os
import sys
import json
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForCausalLM
from pdf2image import convert_from_path

MODEL_ID = "microsoft/Florence-2-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_LABELS = ["title block", "flat pattern", "isometric view", "orthographic view"]
MAX_PAGE_COVERAGE = 0.85  # Exclude canvas bounding boxes (>85% sheet area)
MIN_BOX_AREA = 2000      # Exclude tiny noise artifacts

print("Loading Florence-2 model onto GPU...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    trust_remote_code=True,
    attn_implementation="sdpa" if DEVICE == "cuda" else None
).to(DEVICE)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)


def extract_bounded_cad_regions(image):
    """Executes per-label Phrase Grounding and drops full-canvas bounding boxes."""
    page_area = image.width * image.height
    retained_regions = []

    for label in TARGET_LABELS:
        prompt = f"<CAPTION_TO_PHRASE_GROUNDING> {label}"
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {
            k: v.to(DEVICE, torch.float16) if v.dtype == torch.float32 and DEVICE == "cuda" else v.to(DEVICE)
            for k, v in inputs.items()
        }

        with torch.no_grad():
            generated = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3
            )

        parsed = processor.post_process_generation(
            processor.batch_decode(generated, skip_special_tokens=False)[0],
            task="<CAPTION_TO_PHRASE_GROUNDING>",
            image_size=(image.width, image.height)
        ).get("<CAPTION_TO_PHRASE_GROUNDING>", {})

        bboxes = parsed.get("bboxes", [])
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            area = (x2 - x1) * (y2 - y1)
            coverage = area / page_area

            if area < MIN_BOX_AREA:
                print(f"  -> Skipped '{label}': area {int(area)}px below minimum threshold")
                continue

            if coverage >= MAX_PAGE_COVERAGE:
                print(f"  -> Ignored full-sheet canvas box for '{label}' ({coverage:.1%} coverage)")
                continue

            retained_regions.append({
                "label": label,
                "bbox": bbox,
                "area_px": int(area),
                "page_coverage_pct": round(coverage * 100, 2)
            })

    return retained_regions


def run_pipeline(image_path, output_dir="crops", json_out="drawing_analysis.json"):
    """
    Stage 1: Multi-pass CAD region grounding with spatial coverage filtering.
    Stage 2: Regional crop slicing and OCR feature extraction.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image '{image_path}' not found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)

    print(f"\n[Stage 1] Grounding targeted CAD regions on '{image_path}'...")
    regions = extract_bounded_cad_regions(image)
    print(f"  -> Retained {len(regions)} valid sub-regions after filtering.")

    pipeline_results = []
    print(f"[Stage 2] Running OCR on isolated region crops...")

    for idx, region in enumerate(regions):
        label = region["label"]
        bbox = region["bbox"]
        clean_label = label.strip().replace(" ", "_")

        # Visual bounding box annotation
        draw.rectangle(bbox, outline="red", width=3)
        draw.text((bbox[0], max(0, bbox[1] - 12)), f"{label} ({region['page_coverage_pct']}%)", fill="red")

        # Save region crop
        crop = image.crop(bbox)
        crop_filename = f"crop_{idx}_{clean_label}.png"
        crop_path = os.path.join(output_dir, crop_filename)
        crop.save(crop_path)

        # Stage 2 OCR
        inputs_ocr = processor(text="<OCR_WITH_REGION>", images=crop, return_tensors="pt")
        inputs_ocr = {
            k: v.to(DEVICE, torch.float16) if v.dtype == torch.float32 and DEVICE == "cuda" else v.to(DEVICE)
            for k, v in inputs_ocr.items()
        }

        with torch.no_grad():
            gen_ocr = model.generate(
                input_ids=inputs_ocr["input_ids"],
                pixel_values=inputs_ocr["pixel_values"],
                max_new_tokens=1024,
                num_beams=3
            )

        parsed_ocr = processor.post_process_generation(
            processor.batch_decode(gen_ocr, skip_special_tokens=False)[0],
            task="<OCR_WITH_REGION>",
            image_size=(crop.width, crop.height)
        ).get("<OCR_WITH_REGION>", {})

        pipeline_results.append({
            "crop_id": idx,
            "label": label,
            "bbox": bbox,
            "area_px": region["area_px"],
            "page_coverage_pct": region["page_coverage_pct"],
            "crop_file": crop_path,
            "ocr_text": parsed_ocr.get("labels", []),
            "ocr_regions": parsed_ocr.get("bboxes", [])
        })

    annotated_filename = f"annotated_{os.path.basename(image_path)}"
    annotated_image.save(annotated_filename)
    with open(json_out, "w") as f:
        json.dump(pipeline_results, f, indent=4)

    print(f"\nOutputs Exported:")
    print(f"--> Annotated drawing : {annotated_filename}")
    print(f"--> Isolated crops    : '{output_dir}/' ({len(pipeline_results)} files)")
    print(f"--> Analysis JSON      : {json_out}")


def process_pdf(pdf_path, dpi=300):
    print(f"\nRasterizing PDF '{pdf_path}' at {dpi} DPI...")
    images = convert_from_path(pdf_path, dpi=dpi)

    for i, img in enumerate(images):
        page_filename = f"page_{i+1}.png"
        img.save(page_filename, "PNG")
        print(f"\n=== Processing Page {i+1} of {len(images)} ===")
        run_pipeline(
            image_path=page_filename,
            output_dir=f"crops_page_{i+1}",
            json_out=f"analysis_page_{i+1}.json"
        )


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "reference_samples/001_redacted.pdf"
    if not os.path.exists(target):
        print(f"Error: File '{target}' not found.")
        return

    if target.lower().endswith(".pdf"):
        process_pdf(target)
    else:
        run_pipeline(target)


if __name__ == "__main__":
    main()