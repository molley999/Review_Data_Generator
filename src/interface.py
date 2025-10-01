
import gradio as gr
from src.generation import gradio_stream

def create_demo():
    # Sample products for testing
    sample_products = [
        "WiFi-enabled Smart Toaster",
        "Self-Stirring Coffee Mug",
        "Bluetooth Shower Curtain",
        "AI-Powered Houseplant",
        "Voice-Activated Umbrella"
    ]


    with gr.Blocks(css="""
        .gr-box {border-radius: 10px;}
        .gr-button {background: linear-gradient(90deg, #4F46E5, #6366F1); color: white;}
        .gr-button:hover {opacity: 0.9;}
    """) as demo:

        gr.Markdown(
            "<h1 style='text-align:center;'>🌟 AI Review Generator</h1>"
            "<p style='text-align:center;'>Generate realistic product reviews using multiple AI models.<br>"
            "Perfect for testing e-commerce sites!</p>"
        )

        with gr.Row():
            # Left column (inputs)
            with gr.Column(scale=1):
                # gr.Markdown("### ✍️ Input")
                product = gr.Textbox(
                    label="Product Name",
                    placeholder="Enter any product name...",
                    value="Smart Bluetooth Toothbrush"
                )

                category = gr.Dropdown(
                    choices=["Electronics", "Home & Garden", "Books", "Clothing", "Food & Drink", "Sports", "Toys", "Other"],
                    label="Product Category",
                    value="Electronics"
                )

                num_reviews = gr.Slider(
                    minimum=1, maximum=5, value=3, step=1,
                    label="Number of Reviews"
                )

                include_negative = gr.Checkbox(
                    label="Include negative reviews (1-2 stars)", value=False
                )

                generate_btn = gr.Button("Generate Reviews")

            # Right column (outputs)
            with gr.Column(scale=1):
                # gr.Markdown("### 📤 Output")
                out_text = gr.Textbox(
                    label="Generated Reviews",
                    lines=15,
                    placeholder="Generated reviews will appear here..."
                )
                out_audio = gr.Audio(label="Audio Review")

        # Examples across both columns
        # gr.Markdown("### 💡 Examples")
        gr.Examples(
            examples=[
                ["WiFi-enabled Smart Toaster", "Electronics", 3, False],
                ["Self-Stirring Coffee Mug", "Home & Garden", 4, True],
                ["AI-Powered Houseplant", "Home & Garden", 2, False]
            ],
            inputs=[product, category, num_reviews, include_negative]
        )

        # Button action
        generate_btn.click(
            fn=gradio_stream,
            inputs=[product, category, num_reviews, include_negative],
            outputs=[out_text, out_audio]
        )
        
    return demo