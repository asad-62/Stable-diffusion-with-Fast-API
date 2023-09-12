from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline
from PIL.Image import Image
from transformers import CLIPTextModel, CLIPTokenizer


pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float32)
prompt = "elon musk riding donkey in the middle of snow in hot summer"
image=pipe(prompt)["sample"][0]
def obtain_image(
        prompt: str,
        *,
        seed: int | None = None,
        num_inference_steps: int = 5,
        guidance_scale: float = 7.5   
) -> Image:
    generator = None if seed is None else torch.Generator()
    print(f"using device: {pipe.device}")
    image: Image = pipe(prompt,
                       guidance_scale=guidance_scale,
                       num_inference_steps=num_inference_steps,
                       generator=generator,
                       ).image[0]
    return image

#image=obtain_image(prompt,num_inference_steps=5,seed=1024)