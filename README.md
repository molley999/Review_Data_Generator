# AI Review Generator

Generate realistic product reviews using multiple state-of-the-art language models with real-time streaming and text-to-speech output.

## Overview

This project demonstrates practical LLM engineering patterns including multi-model architectures, memory optimization through quantization, streaming generation, and text-to-speech integration. Perfect for generating synthetic test data for e-commerce platforms, product development, and market research.

## Features

- **Multi-Model Architecture**: Rotates between Phi-3, Llama 3.1, and Gemma-2 for diverse review styles
- **Real-Time Streaming**: Token-by-token generation with immediate UI feedback
- **Text-to-Speech**: Audio output for all generated reviews
- **Memory Efficient**: 4-bit quantization reduces GPU requirements from 40GB+ to 15-20GB
- **Production Patterns**: Model pre-loading, proper error handling, and modular architecture

## Models Used

| Model                | Size | Use Case             | Style               |
| -------------------- | ---- | -------------------- | ------------------- |
| Microsoft Phi-3 Mini | 3.8B | Enthusiastic reviews | Detailed, positive  |
| Meta Llama 3.1       | 8B   | Balanced reviews     | Pros/cons, thorough |
| Google Gemma-2       | 2B   | Casual reviews       | Short, informal     |

## Project Structure

```plaintext
review_generator/
├── app.py # Main entry point
├── config.py # Configuration and constants
├── requirements.txt # Dependencies
├── src/
│ ├── init.py # Package initialization
│ ├── models.py # Model loading and caching
│ ├── prompts.py # Prompt engineering
│ ├── generation.py # Review generation logic
│ ├── tts.py # Text-to-speech
│ └── interface.py # Gradio UI
└── README.md
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU with 16GB+ VRAM (recommended)
- HuggingFace account with API token
- Access to Meta Llama models (requires accepting terms)

### Setup

```bash
# Clone or download the project
git clone <your-repo-url>
cd review_generator

# Install dependencies
pip install -r requirements.txt

# For Google Colab, use the install commands in the notebooks

HuggingFace Authentication

Create account at huggingface.co
Generate API token with read permissions
Accept Meta Llama 3.1 terms at meta-llama/Meta-Llama-3.1-8B-Instruct
In Google Colab: Store token in Secrets as HF_TOKEN

Usage
Quick Start
python app.py

```
