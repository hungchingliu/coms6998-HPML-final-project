# coms6998-HPML-final-project: Analyzing and Optimizing the Performance of Fine Tuning Stable Diﬀusion Image Generating model
- Team Members: Robin Hung-Ching Liu (hl3818)
## TODO: The code is still in the other repository; will migrate it here ASAP.
## Goal/Objective
1. Understand the underlying mechanism and architecture of the Stable Diﬀusion image
generation model.
2. Analyze open-source implementation of fine-tuning Stable Diﬀusion model and eval-
uate its performance under the Google Cloud environment.
3. Apply optimization techniques, including quantization,
LoRA (Low-Rank Adaptation) to accelerate model fine-tuning, inference and improve model performance.
## Challenges
1. Skill Gaps: While I have fundamental knowledge in deep learning theory, I lack
hands-on experience with relevant libraries and frameworks. Bridging this gap may take
some time.
2. Solo Work: While working solo gives me flexibility and more learning, it also means
that I need to handle all aspects of the project, from research to implementation and
evaluation.
## Approach and Performance Optimization Techniques
I plan to explore and experiment the following optimization techniques to enhance
model performance:
- quantization
- LoRA (Low-Rank Adaptation)
## Implementation details
Hardware: Google Cloud Platform VM with GPU(NVIDIA L4)

Software: PyTorch, WandB

Existed code:
- SimpleTuner: https://github.com/bghira/SimpleTuner
- stablediﬀusion (https://github.com/Stability-AI/stablediﬀusion)
- stable-diﬀusion-webui (https://github.com/Stability-AI/stablediﬀusion)
- sd3.5 inference-only (https://github.com/Stability-AI/sd3.5)
- Fine-tuning dataset: paintings by John Singer Sargent
(https://drive.google.com/file/d/1capT9kF-zCu2OiNVzm7VG5DQDaAQLl1Q/view?usp=sharing)
y

## Demo planned
- Weeks 1: Conduct literature review on Stable Diﬀusion and LoRA to understand 
their theoretical foundations and implementations.
- Weeks 2: Set up the development environment, including the necessary software
(PyTorch, WandB, training pipeline) and hardware (GCP with GPU instances).
- Weeks 3: Run experiments to analyze the performance of fine-tuning Stable Diﬀusion
model on GCP using torch.profiler, calculating runtime for diﬀerent stages.
- Weeks 4-5: Apply performance optimization techniques such as torch.compile, mixed
precision, quantization, LoRA (Low-Rank Adaptation), and evaluate the improvements
in model inference speed and quality.
- Weeks 6: Write a report summarizing findings, optimizations, and potential
future improvements.

### References
#### Paper
- SDXL: Improving Latent Diﬀusion Models for High-Resolution Image Synthesis
(https://doi.org/10.48550/arXiv.2307.01952)
- Adding Conditional Control to Text-to-Image Diﬀusion Models
(https://doi.org/10.48550/arXiv.2302.05543)
#### open-source code:
- https://github.com/Stability-AI/stablediﬀusion
- https://github.com/AUTOMATIC1111/stable-diﬀusion-webui
- https://github.com/Stability-AI/sd3.5 (inference-only)
- https://github.com/bghira/SimpleTuner/tree/main
#### Tutorial:
- https://stabilityai.notion.site/Stable-Diﬀusion-3-Medium-Fine-tuning-Tutorial-17f90d-
f74bce4c62a295849f0dc8fb7e
- https://stabilityai.notion.site/Stable-Diﬀusion-3-5-fine-tuning-guide-11a61cdcd1968027a15bdb-
d7c40be8c6