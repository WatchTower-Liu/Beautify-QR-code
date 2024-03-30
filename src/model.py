from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from PIL import Image
import os 
from typing import List
import numpy as np
import torch

from src.QR_code_base_gen import QrCode

class Model:
    def __init__(self, controlnet_path: str, stable_diffusion_path: str):
        self.controlnet_path = controlnet_path if controlnet_path.endswith("/") else controlnet_path + "/"
        self.controlnet_model_path_list, self.controlnet_config_path = self.get_controlnet_file_name()
        self.stable_diffusion_path = stable_diffusion_path if stable_diffusion_path.endswith("/") else stable_diffusion_path + "/"
        self.stable_diffusion_model_path_list, self.stable_diffusion_config_path = self.get_stable_diffusion_file_name()

    def get_model_name(self):
        self.stable_diffusion_model_path_list, self.stable_diffusion_config_path = self.get_stable_diffusion_file_name()
        return self.stable_diffusion_model_path_list
    
    def check_controlnet(self):
        if len(self.controlnet_model_path_list) == 0:
            return False
        return True

    def fresh_controlnet(self):
        self.controlnet_model_path_list, self.controlnet_config_path = self.get_controlnet_file_name()

    def get_stable_diffusion_file_name(self):
        file_list = os.listdir(self.stable_diffusion_path)
        model_path = []
        config_path = []
        for file in file_list:
            if file.endswith(".safetensors"):
                model_path.append(file)
            elif file.endswith(".yaml"):
                config_path.append(self.stable_diffusion_path+file)

        return model_path, config_path[0] if len(config_path) > 0 else None

    def get_controlnet_file_name(self):
        file_list = os.listdir(self.controlnet_path)
        model_path = []
        config_path = []
        for file in file_list:
            if file.endswith(".pth") or file.endswith(".safetensors"):
                model_path.append(file)
            elif file.endswith(".yaml"):
                config_path.append(self.controlnet_path+file)

        return model_path, config_path[0] if len(config_path) > 0 else None

    def load_model(self, stable_diffusion_name, device="cuda:0"):
        print(stable_diffusion_name)
        self.controlnet = ControlNetModel.from_single_file(self.controlnet_path+self.controlnet_model_path_list[0],
                                                           original_config_file=self.controlnet_config_path, 
                                                           local_files_only=True, 
                                                           torch_dtype=torch.float16,
                                                           use_safetensors=True)

        self.pipe = StableDiffusionControlNetPipeline.from_single_file(self.stable_diffusion_path+stable_diffusion_name, 
                                                                       controlnet=self.controlnet, 
                                                                       torch_dtype=torch.float16,
                                                                       local_files_only=True, 
                                                                       original_config_file=self.stable_diffusion_config_path,
                                                                       load_safety_checker=False)
        self.pipe.to(device)

    @staticmethod
    def draw_QR(qrcode: QrCode, hight: int, width: int, scale:float, position_top: int, position_left: int):
        base_map = np.ones([hight, width, 3], dtype=np.uint8) * 255
        draw_map_min_size = min(hight, width)
        code_size = qrcode.get_size()
        code_map = np.ones([code_size, code_size, 3], dtype=np.uint8)*255
        for y in range(code_size):
            for x in range(code_size):
                code_map[y, x, :] = 0 if qrcode.get_module(x,y) else 255

        code_map = Image.fromarray(code_map)
        code_map = code_map.resize([int(draw_map_min_size*scale), int(draw_map_min_size*scale)], Image.NEAREST)
        pos_top_start_legal = position_top if position_top+code_map.size[0] < base_map.shape[0] else base_map.shape[0]-code_map.size[0]
        pos_left_start_legal = position_left if position_left+code_map.size[0] < base_map.shape[1] else base_map.shape[1]-code_map.size[0]
        base_map[pos_top_start_legal:code_map.size[0]+pos_top_start_legal, 
                 pos_left_start_legal:code_map.size[0]+pos_left_start_legal, :] = np.array(code_map)

        return base_map

    def generate_QR(self, QR_string:str, hight: int, width: int, scale:float, position_top: int, position_left: int, placeholder=None):
        errcorlvl = QrCode.Ecc.QUARTILE  # Error correction level
        qr = QrCode.encode_text(QR_string, errcorlvl)
        QR_img = self.draw_QR(qr, hight, width, scale, position_top, position_left)
        return QR_img

    def post_process(self, img, qr_img, value):
        qr_img = qr_img.resize(img.size)
        qr_img = np.array(qr_img)
        img = np.int16(img)
        new_img = np.where(qr_img > 128, np.clip(img/value, 0, 255), img)
        new_img = Image.fromarray(new_img.astype(np.uint8))
        return new_img

    def style_QR(self, pos_prompt, neg_prompt, num_inference_steps, 
                 guidance_scale, controlnet_conditioning_scale, control_guidance_start, control_guidance_end, post_value, qr_img):
        qr_img_PIL = Image.fromarray(qr_img)

        result = self.pipe(pos_prompt, negative_prompt = neg_prompt, image=qr_img_PIL, 
                           controlnet_conditioning_scale = float(controlnet_conditioning_scale),
                           num_inference_steps=num_inference_steps, guidance_scale = guidance_scale,
                           control_guidance_start = control_guidance_start, control_guidance_end = control_guidance_end,
                           clip_skip = 2)
        style_QR = self.post_process(result.images[0], qr_img_PIL, post_value)
        return style_QR

def main():
    pass

if __name__ == "__main__":
    main()
