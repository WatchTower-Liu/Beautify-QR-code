import gradio as gr
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
print(torch.cuda.is_available())
import sys
sys.path.append('../')
from src.model import Model
from src.utils import download_controlnet

def UI():
    gen_model = Model("./models/control_net/", "./models/stable_diffusion/")
    default_model = gen_model.get_model_name()
    if len(default_model) >0 and gen_model.check_controlnet():
        gen_model.load_model(default_model[0])
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Group():
                with gr.Row():
                    model_list = gr.Dropdown(label="SD_model")
                    def ref_b_e():
                        model_list.choices = []
                        for model in gen_model.get_model_name():
                            model_list.choices.append(model)
                        model_list.value = model_list.choices[0]

                        return gr.update(**({"choices": model_list.choices} or {}))

                    ref_b_e()
                    model_list.select(gen_model.load_model, [model_list])
                    refresh_model = gr.Button("refresh model")
                    refresh_model.click(ref_b_e, [], [model_list])

            with gr.Group():
                    with gr.Column():
                        model_help = gr.Markdown("下载模型: SD模型可以从civitai下载, 下载后放置到：./models/stable_diffusion 下，点击下载模型可以下载controlnet\ncontrolnet 状态：{}".format("已下载" if gen_model.check_controlnet() else "未下载"))
                        def download_process():
                            download_controlnet()
                            gen_model.fresh_controlnet()
                            return gr.update(**{"value": "下载模型: SD模型可以从civitai下载, 下载后放置到：./models/stable_diffusion 下，点击下载模型可以下载controlnet\ncontrolnet 状态：{}".format("已下载" if gen_model.check_controlnet() else "未下载")})
                        gr.Button("download controlnet").click(download_process, [], [model_help])
        with gr.Row():
            with gr.Column():
                QR_img = gr.Image(show_label=False)
                QR_text = gr.Textbox(label="Text", placeholder="Enter text here")

                QR_gen_button = gr.Button("generate QR")
                QR_img_higth = gr.Slider(label="QR_img_higth", minimum=256, maximum=2048, step=1, value=768)
                QR_img_width = gr.Slider(label="QR_img_width", minimum=256, maximum=2048, step=1, value=768)
                QR_code_scale = gr.Slider(label="QR_code_scale", minimum=0.1, maximum=1, step=0.01, value=0.8)
                QR_code_position_top = gr.Slider(label="QR_code_position_top", minimum=0, maximum=2048, step=5, value=77)
                QR_code_position_left = gr.Slider(label="QR_code_position_left", minimum=0, maximum=2048, step=5, value=77)
            with gr.Column():
                
                style_img = gr.Image(show_label=False)
                style_pos_prompt = gr.Textbox(label="pos_prompt", placeholder="Enter pos_prompt here")
                style_neg_prompt = gr.Textbox(label="neg_prompt", placeholder="Enter neg_prompt here")

                style_gen_button = gr.Button("generate artQR")

                with gr.Accordion("advance", open=False):
                    style_num_inference_steps = gr.Slider(label="num_inference_steps", minimum=10, maximum=100, step=1, value=50)
                    style_guidance_scale = gr.Slider(label="guidance_scale", minimum=0.5, maximum=15, step=0.5, value=7.5)
                    controlnet_conditioning_scale = gr.Slider(label="controlnet_conditioning_scale", minimum=0, maximum=2.0, step=0.05, value=1.2)
                    control_guidance_start = gr.Slider(label="control_guidance_start", minimum=0, maximum=1, step=0.05, value=0.0)
                    control_guidance_end = gr.Slider(label="control_guidance_end", minimum=0, maximum=1, step=0.05, value=0.8)
                    post_value = gr.Slider(label="post_value", minimum=0.01, maximum=2, step=0.01, value=1.1)
                
        QR_gen_button.click(gen_model.generate_QR, [QR_text,
                                                    QR_img_higth, 
                                                    QR_img_width,
                                                    QR_code_scale,
                                                    QR_code_position_top,
                                                    QR_code_position_left], 
                                                   [QR_img])
        
        style_gen_button.click(gen_model.style_QR, [style_pos_prompt,
                                                    style_neg_prompt,
                                                    style_num_inference_steps,
                                                    style_guidance_scale,
                                                    controlnet_conditioning_scale,
                                                    control_guidance_start,
                                                    control_guidance_end,
                                                    post_value,
                                                    QR_img], 
                                                   [style_img])

    demo.launch(server_port=23200)

if __name__ == "__main__":
    UI()
    
