import torch
from diffusers import DiffusionPipeline
from optimum.quanto import quantize, freeze, qint8
import os

validation_prompts = [
    "a close up three fourth perspective portrait view of a young woman with dark hair and dark blue eyes, looking upwards and to the left, head tilted slightly downwards and to the left, exposed forehead, wearing a nun habit with white lining, wearing a white collared shirt, barely visible ear, cropped, a dark brown background"
]

# Negative prompt
negative_prompt = 'blurry, cropped, ugly'

# Model and adapters
model_id = 'stabilityai/stable-diffusion-3-medium-diffusers'
adapter_ids = [
    'RobinHCL/simpletuner-lora-1',
    'RobinHCL/simpletuner-lora-16',
    'RobinHCL/simpletuner-lora-64',
    'RobinHCL/simpletuner-lora-64-14000',
]

# Set device
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# Load model once
pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
quantize(pipeline.transformer, weights=qint8)
freeze(pipeline.transformer)
pipeline.to(device)

# Loop through adapters and prompts
for adapter_id in adapter_ids:
    # Load LoRA weights
    pipeline.load_lora_weights(adapter_id)

    # Folder for this adapter's outputs
    adapter_name = adapter_id.split('/')[-1]
    output_dir = f"sargent_validation_outputs/{adapter_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Generate images for all prompts
    for idx, prompt in enumerate(validation_prompts):
        generator = torch.Generator(device=device).manual_seed(42)
        image = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=20,
            generator=generator,
            width=1024,
            height=1024,
            guidance_scale=3.0,
        ).images[0]

        filename = os.path.join(output_dir, f"{adapter_name}_{idx+1:02}.png")
        image.save(filename, format="PNG")
        print(f"Saved: {filename}")
