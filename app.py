
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import re

# ================================
# Load Models
# ================================
@st.cache_resource(show_spinner=False)
def load_models():
    # Sentiment Classifier (DistilBERT)
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    
    # GPT-2 Text Generator
    generator_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(generator_name)
    model = AutoModelForCausalLM.from_pretrained(generator_name)
    
    return sentiment_pipeline, tokenizer, model

sentiment_pipeline, tokenizer, model = load_models()

# ================================
# Helper Functions
# ================================
def clean_prompt(text):
    """
    Cleans the user prompt by removing instructions or meta-comments.
    """
    lines = text.split("\n")
    content_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Remove lines with instruction keywords
        if re.search(r"\b(write|be positive|if you|try|exercise|too long|not overly|uplifting|motivational)\b", line, re.IGNORECASE):
            continue
        content_lines.append(line)
    return " ".join(content_lines)

def detect_sentiment(text):
    """
    Detects sentiment using DistilBERT.
    Returns 'positive', 'negative', or 'neutral'.
    """
    result = sentiment_pipeline(text)[0]
    label = result["label"].lower()
    if label == "positive":
        return "positive"
    elif label == "negative":
        return "negative"
    else:
        return "neutral"

def generate_text(prompt, sentiment, max_length=200):
    """
    Generates text aligned with the detected sentiment using GPT-2.
    """
    instructions = {
        "positive": "Write a positive, uplifting paragraph about: ",
        "negative": "Write a negative, sad paragraph about: ",
        "neutral": "Write a neutral, factual paragraph about: "
    }
    final_prompt = instructions.get(sentiment, instructions["neutral"]) + prompt
    input_ids = tokenizer.encode(final_prompt, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=0.7,
        top_p=0.9,
        no_repeat_ngram_size=3,
        do_sample=True
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# ================================
# Streamlit UI
# ================================
st.set_page_config(
    page_title="AI Sentiment-Based Text Generator",
    layout="wide",
    page_icon="🤖"
)

# Custom CSS
st.markdown("""
<style>
.title { color: #4B0082; font-size: 36px; font-weight: bold; text-align: center; }
.subtitle { color: #6A5ACD; font-size: 18px; text-align: center; }
.sentiment { font-weight: bold; color: white; padding: 5px 10px; border-radius: 5px; display: inline-block; }
.positive { background-color: #28a745; }
.negative { background-color: #dc3545; }
.neutral { background-color: #ffc107; color: black; }
.generated-text { background-color: #f0f2f6; color: #000000; padding: 15px; border-radius: 10px; font-size: 16px; line-height: 1.6; white-space: pre-wrap; }
.instructions { color: #444; font-size: 14px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# Page Header
st.markdown('<div class="title">🎯 AI Sentiment-Based Text Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter a prompt and generate sentiment-aligned text instantly</div>', unsafe_allow_html=True)
st.markdown("---")

# User Input
st.subheader("📝 Enter Your Prompt")
st.markdown('<div class="instructions">Instructions or extra guidance will be automatically removed for cleaner output.</div>', unsafe_allow_html=True)
user_prompt = st.text_area("", height=150, placeholder="Type your prompt here...")

st.subheader("🎭 Sentiment Selection (Optional)")
sentiment_choice = st.selectbox("Choose sentiment (or Auto Detect):", ["Auto Detect", "positive", "negative", "neutral"])

st.subheader("📏 Generated Text Length")
max_len = st.slider("Select length (in tokens):", min_value=50, max_value=500, value=200, step=10)

# Generate Button
if st.button("🚀 Generate Text"):
    if not user_prompt.strip():
        st.warning("⚠️ Please enter a prompt before generating text.")
    else:
        with st.spinner("Cleaning prompt, detecting sentiment, and generating text..."):
            cleaned_prompt = clean_prompt(user_prompt)
            sentiment = detect_sentiment(cleaned_prompt) if sentiment_choice=="Auto Detect" else sentiment_choice
            generated = generate_text(cleaned_prompt, sentiment, max_length=max_len)
        
        st.markdown("### 🧠 Detected Sentiment")
        st.markdown(f'<span class="sentiment {sentiment}">{sentiment.upper()}</span>', unsafe_allow_html=True)
        st.markdown("### 📝 Generated Text")
        st.markdown(f'<div class="generated-text">{generated}</div>', unsafe_allow_html=True)
