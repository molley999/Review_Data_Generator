
"""CONFIGURATION FOR REVIEW GENERATOR."""

import torch
from transformers import BitsAndBytesConfig

# MODEL CONFIGURATION
MODELS = {
    'enthusiastic': 'microsoft/Phi-3-mini-4k-instruct',
    'balanced': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'casual': 'google/gemma-2-2b-it'
}

# GENERATION PARAMETERS
max_new_tokens = 150
temperature = 0.7

# QUANTIZATION CONFIG
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

# TEXT-TO-SPEECH model
TTS_MODEL = "facebook/mms-tts-eng"

# CUDA IF APPLICABLE
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

