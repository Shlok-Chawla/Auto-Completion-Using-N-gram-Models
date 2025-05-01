#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Professional N-Gram Auto-Completion Interface
English Language Enhanced Model

A clean, professional UI for the enhanced English N-Gram language model.
Features a modern design focused on usability and performance.
"""

import os
import time
import random
import gradio as gr
import nltk
import numpy as np
from ngram_model import load_model, predict_next_word

# Ensure NLTK tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Model paths
MODEL_PATH = "enhanced_en_counts.txt"
VOCAB_PATH = "enhanced_vocab.txt"

# Professional color palette - Changed to dark mode with light text
COLORS = {
    "primary": "#81e6d9",      # Teal/Cyan for headings
    "secondary": "#4fd1c5",    # Lighter teal for accents
    "accent": "#f687b3",       # Pink accent
    "light": "#e2e8f0",        # Light gray for text
    "dark": "#1a202c",         # Very dark blue/gray for background
    "success": "#68d391",      # Green
    "background": "#2d3748"    # Dark slate background
}

# Custom CSS for a professional dark mode look
CUSTOM_CSS = f"""
.gradio-container {{
    background-color: {COLORS["dark"]};
    color: {COLORS["light"]};
}}

.main-header {{
    color: {COLORS["primary"]};
    text-align: center;
    margin-bottom: 1.5rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid {COLORS["secondary"]};
}}

.prediction-panel {{
    border-radius: 8px;
    background-color: {COLORS["background"]};
    box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    padding: 20px;
    margin-top: 10px;
    border: 1px solid #4a5568;
    color: {COLORS["light"]};
}}

.prediction-word {{
    display: inline-block;
    padding: 8px 16px;
    margin: 5px;
    border-radius: 20px;
    background: {COLORS["secondary"]};
    color: {COLORS["dark"]};
    font-weight: 600;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3);
}}

.completed-sentence {{
    padding: 10px;
    margin-bottom: 8px;
    border-radius: 4px;
    background-color: rgba(74, 85, 104, 0.3);
    border-left: 4px solid {COLORS["secondary"]};
    color: {COLORS["light"]};
}}

.metrics-panel {{
    display: flex;
    justify-content: space-between;
    margin: 15px 0;
}}

.metric-item {{
    flex: 1;
    text-align: center;
    padding: 10px;
    background: {COLORS["background"]};
    border-radius: 8px;
    margin: 0 5px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.3);
    border: 1px solid #4a5568;
    color: {COLORS["light"]};
}}

.metric-item div:first-child {{
    color: #a0aec0;
}}

.example-btn {{
    background-color: rgba(74, 85, 104, 0.3);
    border: 1px solid {COLORS["secondary"]};
    border-radius: 4px;
    margin: 5px;
    padding: 8px 12px;
    color: {COLORS["light"]};
    font-weight: 500;
    cursor: pointer;
}}

.example-btn:hover {{
    background-color: {COLORS["secondary"]};
    color: {COLORS["dark"]};
}}

mark {{
    background-color: rgba(79, 209, 197, 0.3);
    color: {COLORS["primary"]};
    padding: 2px 4px;
    border-radius: 2px;
}}

.footnote {{
    font-size: 0.8rem;
    color: #a0aec0;
    font-style: italic;
}}

.header-subtitle {{
    color: {COLORS["secondary"]};
    font-weight: normal;
    margin-top: -15px;
}}

.error-message {{
    background-color: rgba(229, 62, 62, 0.2);
    color: #fc8181;
    padding: 10px;
    border-radius: 4px;
    border-left: 4px solid {COLORS["accent"]};
    margin-bottom: 15px;
}}

/* Override Gradio's default styles for dark mode */
h1, h2, h3, h4 {{
    color: {COLORS["primary"]} !important;
}}

.gradio-container .prose p, 
.gradio-container .prose ol, 
.gradio-container .prose ul {{
    color: {COLORS["light"]} !important;
}}

/* Fix button text visibility */
button {{
    color: {COLORS["dark"]} !important;
}}

/* Make labels visible */
label {{
    color: {COLORS["light"]} !important;
}}
"""

# Load model at startup with improved error handling
print("Loading enhanced English N-Gram model...")
MODEL_LOADED = False
MODEL_ERROR = None

try:
    # First check if files exist
    if not os.path.exists(MODEL_PATH):
        MODEL_ERROR = f"Model file not found: {MODEL_PATH}"
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    
    if not os.path.exists(VOCAB_PATH):
        MODEL_ERROR = f"Vocabulary file not found: {VOCAB_PATH}"
        raise FileNotFoundError(f"Vocabulary file not found: {VOCAB_PATH}")
        
    # Try loading the files
    n_gram_counts_list, vocabulary = load_model(MODEL_PATH, VOCAB_PATH)
    
    if n_gram_counts_list is None or vocabulary is None:
        MODEL_ERROR = "Model or vocabulary loaded as None"
        raise ValueError("Model or vocabulary loaded as None")
        
    if len(n_gram_counts_list) == 0:
        MODEL_ERROR = "Empty N-gram counts list"
        raise ValueError("Empty N-gram counts list")
        
    if len(vocabulary) == 0:
        MODEL_ERROR = "Empty vocabulary"
        raise ValueError("Empty vocabulary")
    
    print(f"✓ Model loaded successfully: {len(vocabulary)} words, {len(n_gram_counts_list)} n-gram levels")
    MODEL_LOADED = True
    
except Exception as e:
    print(f"✗ Error loading model: {e}")
    if MODEL_ERROR is None:
        MODEL_ERROR = str(e)
    MODEL_LOADED = False

# Example sentences for testing
EXAMPLE_SENTENCES = [
    "I want to go",
    "The weather is",
    "She said that she",
    "I was about",
    "We should consider",
    "The project will be",
    "They have been working on",
    "In the future we might"
]

def format_predictions_html(predictions):
    """Format predictions as HTML with professional styling"""
    html = ""
    for word, prob in predictions:
        if word not in ["<s>", "<e>", "<unk>"]:
            html += f'<div class="prediction-word">{word} <small>({prob:.4f})</small></div>'
    return html

def format_sentences_html(input_text, predictions):
    """Format completed sentences as HTML"""
    html = ""
    for i, (word, _) in enumerate(predictions[:3], 1):
        if word not in ["<s>", "<e>", "<unk>"]:
            html += f'<div class="completed-sentence"><b>{i}.</b> {input_text.strip()} <mark>{word}</mark></div>'
    return html

def calculate_metrics(predictions):
    """Calculate prediction metrics for display"""
    if not predictions:
        return {
            "top_prob": 0,
            "avg_prob": 0,
            "diversity": 0
        }
        
    probs = [prob for _, prob in predictions]
    
    # Calculate metrics
    metrics = {
        "top_prob": max(probs),
        "avg_prob": sum(probs) / len(probs),
        "diversity": 0
    }
    
    # Calculate diversity (normalized entropy)
    if len(probs) > 1:
        normalized_probs = [p/sum(probs) for p in probs]
        entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in normalized_probs)
        max_entropy = np.log2(len(probs))
        metrics["diversity"] = entropy / max_entropy if max_entropy > 0 else 0
        
    return metrics

def format_metrics_html(metrics):
    """Format metrics as HTML"""
    return f"""
    <div class="metrics-panel">
        <div class="metric-item">
            <div style="font-size: 0.9em;">Top Probability</div>
            <div style="font-size: 1.3em; font-weight: bold; color: {COLORS["primary"]};">{metrics["top_prob"]:.4f}</div>
        </div>
        <div class="metric-item">
            <div style="font-size: 0.9em;">Average Probability</div>
            <div style="font-size: 1.3em; font-weight: bold; color: {COLORS["primary"]};">{metrics["avg_prob"]:.4f}</div>
        </div>
        <div class="metric-item">
            <div style="font-size: 0.9em;">Prediction Diversity</div>
            <div style="font-size: 1.3em; font-weight: bold; color: {COLORS["primary"]};">{metrics["diversity"]:.2f}</div>
        </div>
    </div>
    """

def predict(input_text, smoothing=0.5, num_predictions=5):
    """Generate next word predictions with improved error handling"""
    if not MODEL_LOADED:
        error_msg = MODEL_ERROR if MODEL_ERROR else "Unknown model loading error"
        return (
            f"⚠️ Model not loaded correctly: {error_msg}",
            f"<div class='error-message'>Please check that the model files exist and are valid:<br>- {MODEL_PATH}<br>- {VOCAB_PATH}</div>",
            "",
            ""
        )
        
    if not input_text or not input_text.strip():
        return (
            "Please enter some text to get predictions.",
            "<div class='error-message'>Input text cannot be empty</div>",
            "",
            ""
        )
    
    # Start timer for performance tracking
    start_time = time.time()
    
    try:
        # Get predictions
        predictions = predict_next_word(
            input_text=input_text.strip(),
            n_gram_counts_list=n_gram_counts_list,
            vocabulary=vocabulary,
            k=float(smoothing),
            num_predictions=int(num_predictions)
        )
        
        # Track processing time
        processing_time = (time.time() - start_time) * 1000  # milliseconds
        
        # Filter out special tokens
        filtered_predictions = [(word, prob) for word, prob in predictions 
                               if word not in ["<s>", "<e>", "<unk>"]]
        
        if not filtered_predictions:
            return (
                "No valid predictions for this input.",
                "<div class='error-message'>No valid predictions could be generated. Try different input text.</div>",
                "",
                ""
            )
        
        # Format results
        predictions_html = format_predictions_html(filtered_predictions)
        sentences_html = format_sentences_html(input_text, filtered_predictions)
        metrics = calculate_metrics(filtered_predictions)
        metrics_html = format_metrics_html(metrics)
        
        # Status message
        status_message = f"Found {len(filtered_predictions)} predictions in {processing_time:.1f}ms"
        
        return (status_message, predictions_html, sentences_html, metrics_html)
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error making predictions: {e}")
        print(error_details)
        return (
            f"Error: {str(e)}",
            f"<div class='error-message'>An error occurred during prediction:<br>{str(e)}</div>",
            "",
            ""
        )

# Create the interface
with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Soft(), title="WordPredict Pro") as demo:
    # Header
    gr.HTML(
        """
        <div class="main-header">
            <h1>WordPredict Pro</h1>
            <h3 class="header-subtitle">N-Gram Enhanced English Language Model</h3>
        </div>
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            # Input section
            input_text = gr.Textbox(
                placeholder="Type your text here and let AI predict what comes next...",
                label="Input Text",
                lines=3
            )
            
            with gr.Row():
                smoothing_slider = gr.Slider(
                    minimum=0.01,
                    maximum=2.0,
                    step=0.01,
                    value=0.5,
                    label="Smoothing Parameter (k)",
                    info="Controls weight given to unseen word combinations"
                )
                
                predictions_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    step=1,
                    value=5,
                    label="Number of Predictions",
                    info="How many suggestions to generate"
                )
            
            predict_btn = gr.Button("Predict Next Word", variant="primary")
            
            # Examples section
            with gr.Accordion("Example Phrases", open=True):
                example_grid = gr.Markdown()
                
                # Create example buttons dynamically
                example_btns_html = ""
                for example in EXAMPLE_SENTENCES:
                    example_btns_html += f'<button class="example-btn" onclick="document.querySelectorAll(\'[data-testid=textbox]\')[0].value = \'{example}\'; document.querySelectorAll(\'[data-testid=textbox]\')[0].dispatchEvent(new Event(\'input\', {{bubbles: true}}))">{example}</button>'
                example_grid.value = example_btns_html
        
        with gr.Column(scale=3):
            # Status message
            status_output = gr.Markdown()
            
            # Prediction results
            with gr.Group(elem_classes="prediction-panel"):
                gr.Markdown("### Predicted Next Words")
                predictions_output = gr.HTML()
            
            # Completed sentences
            with gr.Group(elem_classes="prediction-panel"):
                gr.Markdown("### Completed Sentences")
                sentences_output = gr.HTML()
            
            # Metrics
            metrics_output = gr.HTML()
    
    # About section
    with gr.Accordion("About WordPredict Pro", open=False):
        gr.Markdown(
            """
            ## About WordPredict Pro
            
            WordPredict Pro uses an enhanced N-gram statistical language model to predict the next word in a sequence based on patterns observed in English text data.
            
            ### Model Details
            
            - **Training Data**: Enhanced with blogs, news, and Twitter content
            - **N-gram Range**: 1 to 5 (unigram through 5-gram models)
            - **Vocabulary Size**: Over 50,000 words
            - **Implementation**: Statistical probability with add-k smoothing
            
            ### How It Works
            
            The model analyzes the context you provide and calculates the probability of each possible next word based on its occurrence patterns in the training data. It uses the following metrics:
            
            - **Top Probability**: Confidence in the highest-ranked prediction
            - **Average Probability**: Overall confidence across all predictions
            - **Diversity**: Variety in the prediction set (higher means more diverse options)
            
            ### Smoothing Parameter
            
            The smoothing parameter (k) controls how much probability is assigned to word combinations that weren't seen in the training data. Higher values give more weight to unseen combinations, while lower values favor combinations seen frequently in the training data.
            """
        )
    
    # Footer
    gr.HTML(
        f"""
        <footer>
            <p>WordPredict Pro - English N-Gram Enhanced Model</p>
            <p class="footnote">Based on statistical patterns in English language data</p>
        </footer>
        """
    )
    
    # Set up event handlers
    predict_btn.click(
        fn=predict,
        inputs=[input_text, smoothing_slider, predictions_slider],
        outputs=[status_output, predictions_output, sentences_output, metrics_output]
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch()