import torch
from safetensors.torch import load_file
import os
import sys
from diffusers import DiffusionPipeline,StableDiffusion3Pipeline

# Validation prompts
validation_prompts = [
    "a close up three fourth perspective portrait view of a young woman with dark hair and dark blue eyes, looking upwards and to the left, head tilted slightly downwards and to the left, exposed forehead, wearing a nun habit with white lining, wearing a white collared shirt, barely visible ear, cropped, a dark brown background"
]

negative_prompt = "blurry, cropped, ugly"
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# Output folders
base_out = "sargent_validation_outputs/base_model"
fullrank_out = "sargent_validation_outputs/fullrank_model"
os.makedirs(base_out, exist_ok=True)
os.makedirs(fullrank_out, exist_ok=True)

# Load base SD3 model
pipeline_base = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16
)
pipeline_base.to(device)

# Inference loop
for idx, prompt in enumerate(validation_prompts):
    print(f"Processing prompt {idx+1}/10")

    generator = torch.Generator(device=device).manual_seed(42 + idx)

    # Base model output
    image_base = pipeline_base(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=20,
        generator=generator,
        width=1024,
        height=1024,
        guidance_scale=3.0,
    ).images[0]
    image_base.save(os.path.join(base_out, f"base_{idx+1:02}.png"))
