import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Set page config as the FIRST command
st.set_page_config(page_title="📰 Fake News Detector", page_icon="📰", layout="centered")

# Load Model & Tokenizer
MODEL_PATH = "nets/BERT.ckpt"
MODEL_NAME = "bert-base-uncased"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

@st.cache_resource
def load_tokenizer():
    return BertTokenizer.from_pretrained(MODEL_NAME)

model = load_model()
tokenizer = load_tokenizer()

# Sidebar for additional information
st.sidebar.title("About")
st.sidebar.info(
    """
    This is a **Fake News Detector** app to classify news articles as **Real** or **Fake**.
    """
)
st.sidebar.markdown("---")
st.sidebar.markdown("### How to Use")
st.sidebar.write("1. Enter the news article text in the text area below.")
st.sidebar.write("2. Click the **Predict** button to see the result.")
st.sidebar.markdown("---")
# st.sidebar.markdown("### Model Information")
# st.sidebar.write(f"Model: **{MODEL_NAME}**")
# st.sidebar.write(f"Device: **{device}**")
# st.sidebar.markdown("---")
st.sidebar.markdown("### Developer")
#st.sidebar.write("Developed by **Anish Pranjal Keshri**<br>(2101020137)", unsafe_allow_html=True)
st.sidebar.write("Developed by **Anish Pranjal Keshri**", unsafe_allow_html=True)

# Main content
st.title("📰 Fake News Detector")
st.write("Enter a news article and check if it's **real** or **fake**.")

# Input text box with placeholder text
user_input = st.text_area(
    label="",  # Empty label since we don't want a label above the text box
    placeholder="Enter News Text :",  # Placeholder text inside the text box
    height=100,
)

# Predict button
if st.button("Predict", key="predict_button"):
    if user_input.strip():
        # Tokenize input
        inputs = tokenizer(user_input, padding="max_length", max_length=512, truncation=True, return_tensors="pt")
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # Get prediction with a progress spinner
        with st.spinner("Analyzing the news article..."):
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                prediction = torch.argmax(logits, dim=1).item()

        # Display result with more visual elements
        st.markdown("---")
        if prediction == 1:
            st.success("✅ **Real News**")
            st.balloons()
        else:
            st.error("❌ **Fake News**")
    else:
        st.warning("⚠️ Please enter some text.")

# Footer
st.markdown("---")
st.markdown("### About the App")
st.write("Welcome to the Fake News Detector! In today’s digital age, misinformation spreads rapidly, making it challenging to distinguish between credible news and fabricated stories. This app is designed to help you quickly assess whether a news article is likely to be real or fake. Using advanced natural language processing techniques, the app analyzes the text of a news article and provides a prediction about its authenticity. Whether you're a journalist, researcher, or just a curious reader, this tool can help you make more informed decisions about the content you consume.")
st.markdown("---")
st.markdown("### Disclaimer")
st.write("This app is for educational purposes only. The predictions are based on a machine learning model and may not always be accurate.")
st.markdown("---")
st.markdown("### Credits")
st.write("Developed with ❤️ by **Anish Pranjal Keshri**")