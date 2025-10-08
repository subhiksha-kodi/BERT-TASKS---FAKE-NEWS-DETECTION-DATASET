import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# ----- Page Config -----
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ----- Custom CSS -----
st.markdown("""
    <style>
        html, body, [class*="css"] {
            background-color: #fdfdfd;
            font-family: 'Segoe UI', sans-serif;
        }

        .title {
            text-align: center;
            color: #4b2e83;
            font-size: 2.8em;
            font-weight: 600;
            margin-bottom: 0.2em;
            letter-spacing: 0.5px;
        }

        .subtitle {
            text-align: center;
            color: #666;
            font-size: 1.1em;
            margin-top: 0;
            margin-bottom: 2rem;
        }

        div.stButton > button:first-child {
            background-color: #4b2e83;
            color: white;
            border: none;
            padding: 0.6em 1.5em;
            font-size: 1em;
            border-radius: 8px;
            font-weight: bold;
            transition: 0.2s ease-in-out;
        }

        div.stButton > button:first-child:hover {
            background-color: #673ab7;
            transform: scale(1.02);
        }

        .prediction-box {
            background-color: #ffffff;
            border-left: 6px solid #4b2e83;
            padding: 1.5rem;
            border-radius: 12px;
            margin-top: 1.5rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
        }

        .result-real {
            color: #2e7d32;
            font-weight: bold;
            font-size: 1.4em;
        }

        .result-fake {
            color: #c62828;
            font-weight: bold;
            font-size: 1.4em;
        }

        .prob-bar {
            height: 18px;
            background-color: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
            margin-top: 8px;
        }

        .fill-bar {
            height: 100%;
            background-color: #4b2e83;
            border-radius: 10px;
        }

        code {
            font-size: 0.9em;
            background-color: #f5f5f5;
            padding: 4px 8px;
            border-radius: 5px;
            display: inline-block;
            margin-top: 12px;
        }
    </style>
""", unsafe_allow_html=True)

# ----- Load Model from Hugging Face Hub -----
@st.cache_resource
def load_model():
    model_repo = "subhiksha-kodi/bert-fake-news-model"
    
    # For private repos, use HF_TOKEN secret
    HF_TOKEN = os.getenv("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(model_repo, use_auth_token=HF_TOKEN)
    model = AutoModelForSequenceClassification.from_pretrained(model_repo, use_auth_token=HF_TOKEN)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# Device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
if device.type == "cpu":
    st.info("⚠️ Running on CPU (may be slower).")

# ----- UI -----
st.markdown("<h1 class='title'>📰 Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Powered by BERT | Check if the news you see is True or Fake.</p>", unsafe_allow_html=True)

user_input = st.text_area(
    "✍ Enter a news headline or short article text:",
    height=150,
    placeholder="e.g. NASA confirms water on the moon..."
)

# ----- Predict -----
if st.button("🔍 Analyze"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(
            user_input,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()

        # Map prediction strictly to True / Fake
        label_mapping = {0: "Fake", 1: "True"}  # adjust if your model has reversed label order
        label_text = label_mapping[pred]
        label_class = "result-real" if label_text == "True" else "result-fake"
        confidence_percent = round(confidence * 100, 2)

        # Show all class probabilities
        prob_texts = []
        for idx, prob in enumerate(probs[0]):
            class_name = label_mapping[idx]
            prob_texts.append(f"{class_name}: {prob.item():.4f}")
        prob_text = ", ".join(prob_texts)

        st.markdown(f"""
            <div class='prediction-box'>
                <h3 class='{label_class}'>Prediction: {label_text}</h3>
                <p><strong>Confidence:</strong> {confidence_percent}%</p>
                <div class="prob-bar">
                    <div class="fill-bar" style="width: {confidence_percent}%;"></div>
                </div>
                <br/>
                <code>Class Probabilities → {prob_text}</code>
            </div>
        """, unsafe_allow_html=True)
