
"""Main application entry point."""

from src.interface import create_demo
from src.models import preload_all_models
import gradio as gr
from google.colab import userdata
from huggingface_hub import login


if __name__ == "__main__":
    hf_token = userdata.get('HF_TOKEN')
    login(hf_token, add_to_git_credential=True)
    preload_all_models()
    demo = create_demo()
    demo.launch(debug=True, share=True)