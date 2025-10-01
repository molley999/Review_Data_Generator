
from typing import Tuple, List, Dict

def create_review_prompt(
    product_name: str, 
    category: str, 
    star_rating: int, 
    review_style: str
    ) -> Tuple[str, str]:
  """Create different prompts based on review style"""
  
  if review_style == 'enthusiastic':
    system_msg = "You are an excited customer who loves sharing detailed positive experiences."
    user_msg = f"Write an {review_style} {star_rating}-star review for {product_name} ({category}). Be specific about features you loved. Keep under 100 words."

  elif review_style == 'balanced':
    system_msg = "You are a thoughtful reviewer who gives honest, balanced feedback."
    user_msg = f"Write a {star_rating}-star review for {product_name} ({category}). Include pros and cons. Be helpful and honest. Keep under 100 words."

  else:
    system_msg = "You are a casual reviewer who writes short, informal reviews."
    user_msg = f"Write a brief {star_rating}-star review for {product_name} ({category}). Use casual language like texting a friend. Keep under 50 words."

  return system_msg, user_msg


def create_messages(
    model_name: str, 
    system_msg: str, 
    user_msg: str
    ) -> List[Dict[str, str]]:
    # Gemma models don't support "system" role
    if "gemma" in model_name.lower():
        return [{"role": "user", "content": system_msg + " " + user_msg}]
    else:
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]