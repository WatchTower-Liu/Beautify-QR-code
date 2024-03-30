import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import subprocess

def download_controlnet():
    hub_name = "monster-labs/control_v1p_sd15_qrcode_monster"
    file_list = "control_v1p_sd15_qrcode_monster.yaml control_v1p_sd15_qrcode_monster.safetensors config.json"
    local_dir = "./models/control_net/"
    subprocess.run(f"huggingface-cli download --resume-download --local-dir-use-symlinks False {hub_name} --include {file_list} config.json --local-dir {local_dir}", shell=True, check=True)