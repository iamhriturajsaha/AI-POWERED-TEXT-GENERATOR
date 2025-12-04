# 🎭AI-Powered Text Generator

An interactive Streamlit web application that detects sentiment in user input and generates contextually appropriate text using transformer models.

## Overview

This project combines sentiment analysis with natural language generation to create an AI writing assistant that produces emotionally consistent text. The application automatically detects whether input text is positive, negative or neutral, then generates new content matching that sentiment.

**Key Capabilities -**
- Uplifting and encouraging positive content.
- Empathetic and emotional negative content.
- Objective and factual neutral content.

The app includes intelligent prompt cleanup to remove instructions and meta-commentary, ensuring clean text generation focused on meaningful content.

## Usage

1. Enter your text or prompt in the input field.
2. The system automatically detects the sentiment.
3. Click "Generate Text" to create sentiment-aligned content.
4. Adjust the token length slider for longer/shorter outputs.
5. View the color-coded sentiment label and generated text.

<p align="center">
  <img src="Streamlit Screenshots/1.png" alt="1" width="1000"/><br>
  <img src="Streamlit Screenshots/2.png" alt="2" width="1000"/><br>
  <img src="Streamlit Screenshots/3.png" alt="3" width="1000"/><br>
  <img src="Streamlit Screenshots/4.png" alt="4" width="1000"/><br>
  <img src="Streamlit Screenshots/5.png" alt="5" width="1000"/><br>
</p>

## Features

### 🔍 Sentiment Detection
- Powered by DistilBERT fine-tuned on SST-2 dataset.
- Classifies text as Positive, Negative or Neutral.
- Fast and accurate sentiment analysis.

### ✍️ Contextual Text Generation
- Uses GPT-2 for coherent text generation.
- Generates content aligned with detected sentiment.
- Customizable output length.

### 🧹 Intelligent Prompt Processing
- Automatically removes instructional content.
- Filters out meta-comments and editing guidelines.
- Extracts core meaningful content for generation.

### 🎨 User-Friendly Interface
- Clean, modern Streamlit design.
- Color-coded sentiment indicators.
- Adjustable generation parameters.
- Real-time results display.

## Technical Stack

### Models
- **Sentiment Analysis -** `distilbert-base-uncased-finetuned-sst-2-english`
- **Text Generation -** `gpt2` (HuggingFace Transformers)

### Dependencies
```
streamlit
transformers
torch
pyngrok
```

### Architecture Flow
```
User Input → Prompt Cleanup → Sentiment Detection → Prompt Engineering → GPT-2 Generation → Display Results
```

## Installation

### Option 1 - Google Colab (Recommended for Quick Testing)

1. **Install dependencies**
```python
!pip install streamlit transformers pyngrok==4.1.1 torch
```

2. **Configure ngrok**
```python
NGROK_AUTH_TOKEN = "your_token_here"
!ngrok authtoken $NGROK_AUTH_TOKEN
```

3. **Launch the app**
```python
!streamlit run app.py & npx localtunnel --port 8501
```

### Option 2 - Local Installation

1. **Clone the repository**
```bash
git clone https://github.com/iamhriturajsaha/AI-POWERED-TEXT-GENERATOR.git
cd AI-POWERED-TEXT-GENERATOR
```

2. **Create and activate virtual environment**
```bash
# macOS/Linux
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

3. **Run the application**
```bash
streamlit run app.py
```

## Technical Details

### Sentiment Classification
The app uses three sentiment-specific prompts for generation -
- **Positive -** "Write a positive and uplifting paragraph about - {topic}"
- **Negative -** "Write a sad and emotional paragraph about - {topic}"
- **Neutral -** "Write a factual and objective paragraph about - {topic}"

### Generation Parameters
```python
max_length=150
temperature=0.7
top_p=0.9
no_repeat_ngram_size=3
```
These settings balance creativity with coherence while minimizing repetition.

## Known Limitations & Solutions

### Text Repetition
**Issue -** GPT-2 occasionally generates repetitive phrases.  
**Solution -** Implemented `no_repeat_ngram_size=3` and tuned temperature parameters.

### Mixed Sentiment Detection
**Issue -** Complex emotions may confuse the classifier.  
**Solution -** Added preprocessing to clean ambiguous or instructional content.

### Display Formatting
**Issue -** Generated text visibility issues with certain themes.  
**Solution -** Custom CSS classes ensure proper text contrast.

## Future Enhancements

- Support for additional models (GPT-Neo, GPT-J).
- Emotional intensity adjustment slider.
- Multilingual sentiment detection and generation.
- Export generated text to file.
- Dark mode theme toggle.
- Fine-tuned sentiment-specific generation models.
- Batch processing capability.

