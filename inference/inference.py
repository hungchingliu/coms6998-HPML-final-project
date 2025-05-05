import torch
from diffusers import DiffusionPipeline

model_id = 'stabilityai/stable-diffusion-3-medium-diffusers'
adapter_id = 'RobinHCL/simpletuner-lora'
pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16) # loading directly in bf16
pipeline.load_lora_weights(adapter_id)

prompt = "a close up three fourth perspective portrait view of a young woman with dark hair and dark blue eyes, looking upwards and to the left, head tilted slightly downwards and to the left, exposed forehead, wearing a nun habit with white lining, wearing a white collared shirt, barely visible ear, cropped, a dark brown background"
negative_prompt = 'blurry, cropped, ugly'

## Optional: quantise the model to save on vram.
## Note: The model was quantised during training, and so it is recommended to do the same during inference time.
from optimum.quanto import quantize, freeze, qint8
quantize(pipeline.transformer, weights=qint8)
freeze(pipeline.transformer)
    
pipeline.to('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu') # the pipeline is already in its target precision level
model_output = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=20,
    generator=torch.Generator(device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu').manual_seed(42),
    width=1024,
    height=1024,
    guidance_scale=3.0,
).images[0]

model_output.save("output.png", format="PNG")
