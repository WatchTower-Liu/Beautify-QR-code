import sys
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from PIL import Image

url = "F:/huggingface_model/controlnet/controlnetQRPatternQR_v2Sd15.safetensors"  # can also be a local path
controlnet = ControlNetModel.from_single_file(url, local_files_only=True, config_file="D:/code/QR_code_gen/models/control_net/control_v11f1e_sd15_tile.yaml")

url = "F:/SD/stable-diffusion-webui/models/Stable-diffusion/dreamshaper_8.safetensors"  # can also be a local path
pipe = StableDiffusionControlNetPipeline.from_single_file(url, controlnet=controlnet, local_files_only=True, original_config_file="D:/code/QR_code_gen/models/stable_diffusion/v1-inference.yaml", cache_dir = "D:/code/QR_code_gen/models/stable_diffusion/", load_safety_checker=False)
pipe.to("cuda")

img = Image.open("D:/code/QR_code_gen/test/1.png")
img = img.resize((512, 512), Image.NEAREST)

result = pipe("mountain, sky, leake, house, garden with tree", image=img, num_inference_steps=50, 
              control_guidance_start = 0.0, control_guidance_end = 0.6)
result.images[0].show()