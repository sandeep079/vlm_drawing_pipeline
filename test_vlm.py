import torch
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModelForCausalLM

model_id = "microsoft/Florence-2-base"

print("Loading Florence-2 model onto GPU using PyTorch SDPA...")
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    trust_remote_code=True,
    attn_implementation="sdpa"
).to("cuda")

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# Sample image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

# Task prompt
prompt = "<MORE_DETAILED_CAPTION>"
inputs = processor(text=prompt, images=image, return_tensors="pt")
inputs = {k: v.to("cuda", torch.float16) if v.dtype == torch.float32 else v.to("cuda") for k, v in inputs.items()}

with torch.no_grad():
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )

generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
parsed_answer = processor.post_process_generation(generated_text, task=prompt, image_size=(image.width, image.height))

print("\nModel Output:", parsed_answer)
print(f"VRAM Allocated: {round(torch.cuda.memory_allocated()/1e6, 2)} MB")
