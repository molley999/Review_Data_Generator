
from transformers import pipeline
from config import TTS_MODEL, DEVICE
import numpy as np
import os
import soundfile as sf
import tempfile

# GLOBAL: load once (outside gradio_stream to save time)
tts_pipe = pipeline("text-to-speech", model=TTS_MODEL, device=DEVICE)

def text_to_speech(text: str) -> str:
    speech = tts_pipe(text)

    # Ensure audio is 1D
    audio = np.array(speech["audio"]).squeeze()

    # Save as 16-bit PCM WAV
    tmp_wav = os.path.join(tempfile.gettempdir(), "review.wav")
    sf.write(tmp_wav, audio, speech["sampling_rate"], subtype="PCM_16")
    
    return tmp_wav