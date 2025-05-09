import torch
import copy
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
fullrank_out = "sargent_validation_outputs/fullrank_model"
os.makedirs(fullrank_out, exist_ok=True)

# Load base SD3 model
pipeline_fullrank = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16
)
safetensor_path = "/home/robin/HPML/coms6998-HPML-final-project/SimpleTuner/output/models/fulltraining/pipeline/transformer/diffusion_pytorch_model.safetensors"
state_dict = load_file(safetensor_path)
pipeline_fullrank.transformer.load_state_dict(state_dict)
pipeline_fullrank.to(device)

# Inference loop
for idx, prompt in enumerate(validation_prompts):
    print(f"Processing prompt {idx+1}/10")

    generator = torch.Generator(device=device).manual_seed(42 + idx)

    # fullrank model output
    image_fullrank = pipeline_fullrank(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=20,
        generator=generator,
        width=1024,
        height=1024,
        guidance_scale=3.0,
    ).images[0]
    image_fullrank.save(os.path.join(fullrank_out, f"base_{idx+1:02}.png"))
