"""Review generation logic."""

from typing import Generator, Optional, Tuple
from threading import Thread
from transformers import TextIteratorStreamer
import random
import torch

from config import max_new_tokens, temperature, MODELS, DEVICE
from src.models import load_model
from src.tts import text_to_speech
from src.prompts import create_review_prompt, create_messages


def gradio_stream(
    product_name: str, 
    category: str, 
    num_reviews: int, 
    include_ratings: bool
    ) -> Generator[Tuple[str, Optional[str]], None, None]:
    """Main function called by Gradio interface"""

    if not product_name.strip():
      yield "Please enter a product name!", None
      return

    styles = ['enthusiastic', 'balanced', 'casual']
    all_reviews = ""
    
    final_audio = None

    for i in range(int(num_reviews)):
      if include_ratings:
        star_rating = random.choices([5, 4, 3, 2, 1], weights=[40, 30, 15, 10, 5])[0]
      else:
        star_rating = random.choices([5, 4, 3], weights=[50, 30, 20])[0]

      style = styles[i % len(styles)]
      stars = "⭐" * star_rating

      try:
        review_text = ""
        for word in stream_review(product_name, category, star_rating, style):
          review_text += word
          current_entry = f"{stars} ({star_rating}/5)\n{review_text}\n"
          if all_reviews:
            yield all_reviews + "\n" + "="*50 + "\n" + current_entry, None
          else:
            yield current_entry, None

        if all_reviews:
          all_reviews += "\n" + "="*50 + "\n" + current_entry
        else:
          all_reviews = current_entry

      except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        all_reviews += f"\n{error_msg}\n"
        yield all_reviews, None


    if all_reviews.strip():
        tmp_wav = text_to_speech(all_reviews)
        yield all_reviews, tmp_wav
    else:
        yield "❌ No review generated", None
        
        
def stream_review(
    product_name: str, 
    category: str, 
    star_rating: int, 
    review_style: str
    ) -> Generator[str, None, None]:
    """Generate one review using specified model"""

    model, tokenizer = load_model(review_style)

    system_msg, user_msg = create_review_prompt(product_name, category, star_rating, review_style)

    messages = create_messages(MODELS[review_style], system_msg, user_msg)

    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        padding=True,
        add_generation_prompt=True,
        truncation=True
        ).to(DEVICE)


    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = {
        "input_ids": inputs,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": True,
        "streamer": streamer
    }

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Yield tokens as they're generated
    for new_text in streamer:
        yield new_text

    thread.join()