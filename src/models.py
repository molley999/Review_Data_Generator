"""MODEL LOADING AND MANAGEMENT"""

import torch
from typing import Dict, Tuple
from config import MODELS, quant_config
from transformers import AutoTokenizer, AutoModelForCausalLM

# Global cache for loaded models
LOADED_MODELS: Dict[str, Dict] = {}

def preload_all_models() -> Dict[str, Dict]:
    """Pre-load all models at startup."""
    global LOADED_MODELS
    for model_name, model_path in MODELS.items():
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map = "auto",
            quantization_config=quant_config
            )

        LOADED_MODELS[model_name] = {
            'model': model,
            'tokenizer': tokenizer
        }
        
    return LOADED_MODELS

def load_model(model_name: str) -> Tuple:
    """Retrieve pre-loaded model from cache"""
    return LOADED_MODELS[model_name]['model'], LOADED_MODELS[model_name]['tokenizer']
