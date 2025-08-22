import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline

# Title of the Streamlit app
st.title("Fine-Tuned BERT: Multi-Class Sentiment Classifier for Tweets")

# Load the fine-tuned model using Hugging Face pipeline
# 'text-classification' automatically handles preprocessing + prediction

# Cache the model so it doesn't reload every time
@st.cache_resource
def load_model():
    return pipeline('text-classification', model='bert-base-uncased-sentiment-model')

classifier = load_model()
# Input text area for user to type a tweet or text
text = st.text_area("Enter a tweet or any text for sentiment analysis")

# When the "Predict" button is clicked
if st.button("Predict"):
    if text.strip():  # Check if text is not empty
        # Run the classifier
        result = classifier(text, truncation=True)  # truncation avoids long input errors

        # Show prediction nicely
        st.subheader("Prediction Result:")
        for res in result:
            st.write(f"**Label:** {res['label']}, **Score:** {res['score']:.4f}")
    else:
        st.warning("⚠️ Please enter some text before clicking Predict.")
