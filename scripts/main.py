import os
import sys
import json
import gradio as gr
import modules.ui
from modules import script_callbacks, sd_models, shared
from modules.ui import setup_progressbar, gr_show
from modules.shared import opts, cmd_opts, state
from webui import wrap_gradio_gpu_call
from modules.ui import setup_progressbar, gr_show, wrap_gradio_call, create_refresh_button
from extensions.sd_dreambooth_extension.dreambooth.utils import get_lora_models
from lora_diffusion import cli_lora_add

def on_ui_tabs():
    with gr.Blocks() as lora_merger:
        with gr.Column(varint="panel"):
            with gr.Row():
                lora_model_1 = gr.Dropdown(label='Lora Model (A)', choices=sorted(get_lora_models()),interactive=True)
                create_refresh_button(lora_model_1, get_lora_models, lambda: {
                    "choices": sorted(get_lora_models())},
                    "refresh_lora_models")
            with gr.Row():
                lora_model_2 = gr.Dropdown(label='Lora Model (B)', choices=sorted(get_lora_models()),interactive=True)
                create_refresh_button(lora_model_2, get_lora_models, lambda: {
                    "choices": sorted(get_lora_models())},
                    "refresh_lora_models")
            with gr.Row():
                lora_weight = gr.Slider(label="Lora Weight", value=1, minimum=0.1, maximum=1, step=0.1)
            with gr.Row():
                lora_new_model = gr.Textbox(label="Custom Name (Optional)")
            with gr.Row():
                run_button = gr.Button(value="Run", elem_id="run_button", variant='primary')
            
            run_button.click(
                fn=merge,
                inputs=[lora_model_1,lora_model_2,lora_weight,lora_new_model],
                outputs=[],
            )

    return (lora_merger, "Lora Merger", "lora_merger"),

script_callbacks.on_ui_tabs(on_ui_tabs)

def merge(name1,name2,weight,name3):
    path1 =  os.path.join('models', 'lora', name1)
    path2 =  os.path.join('models', 'lora', name2)
    if not name3:
        name3 = name1.split('.')[0] + '_' + name2.split('.')[0] + '_' + str(weight)
    path3 =  os.path.join('models', 'lora', name3+'.pt')

    cli_lora_add.add(path1,path2,path3,weight)
    print("Saved " + path3)
