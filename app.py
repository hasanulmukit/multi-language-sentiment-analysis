import streamlit as st
from transformers import pipeline
import matplotlib.pyplot as plt
import pandas as pd
import speech_recognition as sr
from googletrans import Translator
import io

st.set_page_config(page_title="Sentiment Analysis", page_icon="📝", layout="centered")

# Load Hugging Face Sentiment Analysis Model
@st.cache_resource
def load_model():
    return pipeline(sentiment-analysis)

nlp = load_model()

# Initialize Translator
translator = Translator()

# Streamlit UI
st.title("📝 Sentiment Analysis")
st.markdown("Analyze the sentiment of any text using AI!")

# 🎨 Custom Styling
st.markdown("""
    <style>
        .stTextArea, .stTextInput {
            border-radius: 10px;
        }
        .css-1aumxhk {
            background-color: #f8f9fa !important;
        }
    </style>
""", unsafe_allow_html=True)

# Session State for Sentiment History
if "sentiment_history" not in st.session_state:
    st.session_state.sentiment_history = []

# 🌍 Multi-language Support
language = st.selectbox("Select language:", ["English", "French", "Spanish", "German", "Bengali"])

# 🎙️ Voice Input Support
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        st.success(f"Recognized Text: {text}")
        return text
    except sr.UnknownValueError:
        st.error("Sorry, could not understand the audio.")
    except sr.RequestError:
        st.error("Could not request results. Check your internet connection.")
    return ""

user_text = st.text_area("Enter text:", placeholder="Type something...", height=150)

# 📂 File Upload Feature
uploaded_file = st.file_uploader("Upload a file (.txt or .csv)", type=["txt", "csv"])

if uploaded_file is not None:
    if uploaded_file.type == "text/plain":
        file_text = uploaded_file.read().decode("utf-8")
        st.text_area("File Content", file_text, height=150)
        user_text = file_text  # Use uploaded file content
    elif uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
        if "text" in df.columns:
            user_text = "\n".join(df["text"].tolist())
            st.text_area("File Content", user_text, height=150)
        else:
            st.error("CSV must contain a 'text' column.")

if st.button("Analyze Sentiment"):
    if user_text:
        # Translate to English if not in English
        if language != "English":
            user_text = translator.translate(user_text, src=language.lower(), dest="en").text
            st.info(f"Translated Text: {user_text}")

        sentences = user_text.split(".")  # Split into individual sentences
        results = nlp(sentences)

        # Extract Sentiments
        sentiments = [res["label"] for res in results]
        scores = [res["score"] for res in results]

        # Count Sentiment Distribution
        sentiment_counts = pd.Series(sentiments).value_counts()

        # Display Individual Sentence Results
        st.subheader("Results")
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                sentiment_emoji = "😊" if results[i]["label"] == "POSITIVE" else "😡" if results[i]["label"] == "NEGATIVE" else "😐"
                st.write(f"**Sentence {i+1}:** {sentence.strip()}")
                st.write(f"**Sentiment:** {results[i]['label']} {sentiment_emoji} | **Confidence:** {results[i]['score']:.2%} 🔥")
                st.markdown("---")

                # Store in session state
                st.session_state.sentiment_history.append([sentence.strip(), results[i]['label'], results[i]['score']])

        # 📊 Sentiment Distribution Chart
        st.subheader("📊 Sentiment Distribution")
        fig, ax = plt.subplots()
        sentiment_counts.plot(kind="bar", color=["green", "red", "gray"], ax=ax)
        plt.xlabel("Sentiment")
        plt.ylabel("Count")
        plt.title("Sentiment Analysis Distribution")
        st.pyplot(fig)

        # 📌 Save & Download Results
        df_results = pd.DataFrame({"Sentence": sentences, "Sentiment": sentiments, "Confidence": scores})
        csv = df_results.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Results", data=csv, file_name="sentiment_results.csv", mime="text/csv")

    else:
        st.warning("Please enter some text or upload a file.")

# 📜 Sentiment History
if st.session_state.sentiment_history:
    st.subheader("📜 Sentiment History")
    df_history = pd.DataFrame(st.session_state.sentiment_history, columns=["Sentence", "Sentiment", "Confidence"])
    st.table(df_history)

# Footer
st.markdown("---")
st.markdown("🚀 Built by [Hasanul Mukit](https://github.com/hasanulmukit)")
