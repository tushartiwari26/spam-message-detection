import streamlit as st
import pickle
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "spam_model.pkl"

st.set_page_config(
    page_title="Spam Detector",
    page_icon="📩",
    layout="centered",
)


@st.cache_resource
def load_model():
    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)


st.title("📩 Spam Message Detector")

st.write(
    "Enter a message below to check whether "
    "our machine learning model identifies it as spam."
)

if not MODEL_PATH.is_file():
    st.error(
        "Trained model not found. "
        "Run main.py first to train and save the model."
    )
    st.stop()

model = load_model()

message = st.text_area(
    "Enter your message:",
    placeholder="Example: Congratulations! You won a prize!",
)

if st.button("Analyze Message", type="primary"):

    if not message.strip():
        st.warning("Please enter a message first.")

    else:
        prediction = model.predict([message])[0]
        probabilities = model.predict_proba([message])[0]

        confidence = probabilities[prediction] * 100

        if prediction == 1:
            st.error("🚨 Prediction: SPAM")
        else:
            st.success("✅ Prediction: NOT SPAM")

        st.metric(
            "Model confidence",
            f"{confidence:.2f}%",
        )

        st.caption(
            "Confidence reflects the model's estimated "
            "probability for its prediction, not a guarantee "
            "that the prediction is correct."
        )

st.divider()

st.subheader("How does it work?")

st.write(
    """
    1. The message is converted into numerical features using TF-IDF.
    2. A Naive Bayes classifier analyzes those features.
    3. The model predicts whether the message resembles spam.
    """
)

st.info(
    "This is an educational machine learning project. "
    "The model can make incorrect predictions."
)
