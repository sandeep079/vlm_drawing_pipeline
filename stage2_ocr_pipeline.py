import os
import json
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

model_id = "microsoft/Florence-2-base"
print("Loading Florence-2 for Stage 2 OCR...")
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    trust_remote_code=True,
    attn_implementation="sdpa"
).to("cuda")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

def process_crops_for_ocr(crops_dir="crops", output_json="drawing_analysis.json"):
    if not os.path.exists(crops_dir):
        print(f"Directory '{crops_dir}' not found.")
        return

    crop_files = [f for f in os.listdir(crops_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not crop_files:
        print("No crop images found to process.")
        return

    results = {}

    for crop_file in sorted(crop_files):
        crop_path = os.path.join(crops_dir, crop_file)
        image = Image.open(crop_path).convert("RGB")
        
        task_prompt = "<OCR_WITH_REGION>"
        inputs = processor(text=task_prompt, images=image, return_tensors="pt")
        inputs = {k: v.to("cuda", torch.float16) if v.dtype == torch.float32 else v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3
            )

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
        
        results[crop_file] = parsed_answer.get(task_prompt, {})
        print(f"Processed {crop_file} -> Text detected: {parsed_answer.get(task_prompt, {}).get('labels', [])}")

    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\nStage 2 complete! Summary exported to: {output_json}")

if __name__ == "__main__":
    process_crops_for_ocr()
