import streamlit as st
import joblib
import re
import os
import nltk
from nltk.corpus import stopwords

# =====================================
# Download Stopwords
# =====================================
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# =====================================
# Page Configuration
# =====================================
st.set_page_config(
    page_title="Amazon Sentiment Analysis",
    page_icon="🛒",
    layout="wide"
)

# =====================================
# Custom CSS Styling
# =====================================
st.markdown("""
<style>
/* Main app background */
.stApp {
    background: linear-gradient(to bottom right, #f9fbff, #fef6ff);
    color: #1f2937;
}

/* Remove top whitespace */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Header card */
.header-box {
    background: linear-gradient(135deg, #4f46e5, #7c3aed, #ec4899);
    padding: 28px;
    border-radius: 22px;
    color: white;
    box-shadow: 0 10px 30px rgba(0,0,0,0.12);
    margin-bottom: 20px;
}
.header-title {
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 6px;
}
.header-subtitle {
    font-size: 1.05rem;
    opacity: 0.95;
}

/* Section cards */
.card {
    background: white;
    padding: 22px;
    border-radius: 20px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

/* Result card */
.result-card {
    background: linear-gradient(135deg, #fffdf6, #fff8e7);
    padding: 22px;
    border-radius: 20px;
    border: 1px solid #fde68a;
    box-shadow: 0 8px 25px rgba(0,0,0,0.06);
}

/* Sidebar style */
section[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #ffffff, #f7f4ff);
    border-right: 1px solid #ede9fe;
}
.sidebar-title {
    font-size: 1.8rem;
    font-weight: 800;
    color: #1e3a8a;
    margin-bottom: 0.2rem;
}
.sidebar-sub {
    color: #7c3aed;
    font-size: 1.05rem;
    font-weight: 700;
    margin-bottom: 1rem;
}
.sidebar-card {
    background: white;
    border-radius: 16px;
    padding: 16px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.06);
    margin-bottom: 16px;
}
.sentiment-positive {
    background: #ecfdf5;
    border: 1px solid #86efac;
    border-radius: 14px;
    padding: 12px;
    margin-bottom: 10px;
}
.sentiment-negative {
    background: #fef2f2;
    border: 1px solid #fca5a5;
    border-radius: 14px;
    padding: 12px;
    margin-bottom: 10px;
}
.sentiment-neutral {
    background: #fffbea;
    border: 1px solid #fcd34d;
    border-radius: 14px;
    padding: 12px;
    margin-bottom: 10px;
}

/* Button */
div.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #9333ea);
    color: white;
    font-weight: 700;
    border: none;
    border-radius: 12px;
    padding: 0.7rem 1.4rem;
    font-size: 1rem;
    box-shadow: 0 8px 20px rgba(124,58,237,0.3);
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #6d28d9, #7e22ce);
    color: white;
}

/* Text area */
textarea {
    border-radius: 16px !important;
    border: 2px solid #d8b4fe !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

/* Footer */
.footer-box {
    margin-top: 25px;
    background: linear-gradient(135deg, #ede9fe, #dbeafe);
    padding: 20px;
    border-radius: 18px;
    text-align: center;
    font-size: 1rem;
    color: #1e3a8a;
    box-shadow: 0 6px 20px rgba(0,0,0,0.05);
}
.small-note {
    color: #6b7280;
    font-size: 0.95rem;
}

/* Hide streamlit menu/footer if you want cleaner UI */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =====================================
# Correct Project Paths
# =====================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

MODEL_PATH = os.path.join(MODEL_DIR, "best_sentiment_model.pkl")
TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
LABEL_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

# =====================================
# Load Saved Files
# =====================================
@st.cache_resource
def load_files():
    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(TFIDF_PATH)
    label_encoder = joblib.load(LABEL_PATH)
    return model, tfidf, label_encoder

try:
    model, tfidf, label_encoder = load_files()
except Exception as e:
    st.error(f"Error loading model files:\n\n{e}")
    st.stop()

# =====================================
# Text Cleaning
# =====================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    words = [word for word in text.split() if word not in stop_words]
    return " ".join(words)

# =====================================
# Sidebar
# =====================================
with st.sidebar:
    st.markdown('<div class="sidebar-title">🛒 Amazon</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Sentiment Analysis</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-card">
        <h4>ℹ️ About This App</h4>
        <p>
        This app uses a machine learning model trained on Amazon product reviews
        to predict whether a review is <b>Positive</b>, <b>Negative</b>, or <b>Neutral</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h4>📘 Sentiment Guide</h4>", unsafe_allow_html=True)

    st.markdown("""
    <div class="sentiment-positive">
        <b>😊 Positive</b><br>
        Reviews expressing satisfaction, happiness, or appreciation.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sentiment-negative">
        <b>😞 Negative</b><br>
        Reviews expressing dissatisfaction, complaints, or bad experiences.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sentiment-neutral">
        <b>😐 Neutral</b><br>
        Reviews that are neutral or neither clearly good nor bad.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-card">
        <h4>💻 Technologies Used</h4>
        <ul>
            <li>Python</li>
            <li>Streamlit</li>
            <li>Scikit-learn</li>
            <li>TF-IDF Vectorizer</li>
            <li>Machine Learning Classification Model</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-card" style="text-align:center;">
        <b>⭐ Developed with care</b><br><br>
        Amazon Sentiment Analysis Project
    </div>
    """, unsafe_allow_html=True)

# =====================================
# Main Header
# =====================================
st.markdown("""
<div class="header-box">
    <div class="header-title">🛒 Amazon Product Sentiment Analysis</div>
    <div class="header-subtitle">
        Understand customer opinions and classify product reviews into Positive, Negative, or Neutral sentiment.
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================
# Review Input Section
# =====================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("💬 Enter an Amazon Product Review")

review = st.text_area(
    "",
    height=180,
    placeholder="Example: This product is amazing and worth every penny!"
)

predict_btn = st.button("✨ Predict Sentiment")
st.markdown('</div>', unsafe_allow_html=True)

# =====================================
# Prediction Section
# =====================================
if predict_btn:
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        try:
            cleaned_review = clean_text(review)

            if cleaned_review.strip() == "":
                st.warning("Please enter a meaningful review.")
            else:
                review_vector = tfidf.transform([cleaned_review])

                prediction = model.predict(review_vector)[0]
                sentiment = label_encoder.inverse_transform([prediction])[0]

                confidence = None
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(review_vector)[0]
                    confidence = max(probabilities) * 100

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("📊 Prediction Result")

                if sentiment == "Positive":
                    st.success("😊 Positive Sentiment")
                    sentiment_desc = "The review expresses satisfaction, appreciation, or a good experience."
                elif sentiment == "Negative":
                    st.error("😞 Negative Sentiment")
                    sentiment_desc = "The review expresses dissatisfaction, complaints, or a poor experience."
                elif sentiment == "Neutral":
                    st.warning("😐 Neutral Sentiment")
                    sentiment_desc = "The review is balanced or does not strongly express positive or negative emotion."
                else:
                    st.info(f"Predicted Sentiment: {sentiment}")
                    sentiment_desc = "Predicted sentiment generated by the model."

                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown(f"""
                    <div class="result-card">
                        <h2 style="margin-bottom:8px;">{sentiment} Sentiment</h2>
                        <p class="small-note">{sentiment_desc}</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    if confidence is not None:
                        st.metric("Confidence", f"{confidence:.2f}%")

                with st.expander("🔍 Show cleaned review text"):
                    st.write(cleaned_review)

                st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction Error:\n\n{e}")

# =====================================
# Footer
# =====================================
st.markdown("""
<div class="footer-box">
    <h3 style="margin-bottom:8px;">💜 Thank you for using Amazon Sentiment Analysis!</h3>
    <p>Your feedback helps improve customer experience and review understanding.</p>
</div>
""", unsafe_allow_html=True)