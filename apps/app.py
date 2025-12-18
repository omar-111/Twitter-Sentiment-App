"""
Streamlit web application for Twitter Sentiment Classification.
"""
import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import load_predictor
import pandas as pd

# Page config
st.set_page_config(
    page_title="Twitter Sentiment Classifier",
    page_icon="🐦",
    layout="wide"
)

# Title
st.title("🐦 Twitter Sentiment Classification")
st.markdown("Classify sentiment of tweets using trained ML models")

# Sidebar for model selection
st.sidebar.header("Model Selection")
model_type = st.sidebar.selectbox(
    "Choose Model",
    ["baseline", "rnn"],
    help="Baseline: TF-IDF + LinearSVC\nRNN: Embedding + BiLSTM/GRU"
)

# Load predictor
@st.cache_resource
def load_model(model_type):
    """Load model with caching."""
    try:
        return load_predictor(model_type=model_type)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please train the models first using:")
        st.code(f"python -m src.train_{model_type}")
        return None

predictor = load_model(model_type)

if predictor is not None:
    # Main input area
    st.header("Input")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        tweet_text = st.text_area(
            "Tweet Text",
            placeholder="Enter your tweet here...",
            height=150,
            help="Enter the tweet text you want to classify"
        )
    
    with col2:
        entity = st.text_input(
            "Entity (Optional)",
            placeholder="e.g., Twitter, Google",
            help="Optional: Specify the entity this tweet is about"
        )
    
    # Predict button
    if st.button("🔍 Predict Sentiment", type="primary", use_container_width=True):
        if tweet_text.strip():
            with st.spinner("Analyzing sentiment..."):
                try:
                    result = predictor.predict(tweet_text, entity, return_proba=True)
                    
                    # Display results
                    st.header("Prediction Results")
                    
                    # Main result card
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.metric(
                            "Predicted Sentiment",
                            result["label"],
                            delta=f"{result['confidence']:.1%} confidence"
                        )
                    
                    with col2:
                        st.metric(
                            "Confidence",
                            f"{result['confidence']:.1%}",
                            help="Confidence score for the prediction"
                        )
                    
                    # Probability breakdown
                    st.subheader("Probability Breakdown")
                    prob_df = pd.DataFrame(
                        list(result["probabilities"].items()),
                        columns=["Sentiment", "Probability"]
                    )
                    prob_df = prob_df.sort_values("Probability", ascending=False)
                    prob_df["Probability"] = prob_df["Probability"].apply(lambda x: f"{x:.1%}")
                    
                    st.dataframe(prob_df, use_container_width=True, hide_index=True)
                    
                    # Visual bar chart
                    st.bar_chart(
                        {k: v for k, v in result["probabilities"].items()},
                        height=300
                    )
                    
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
                    st.exception(e)
        else:
            st.warning("Please enter a tweet text to classify.")
    
    # Example section
    with st.expander("📝 Example Tweets"):
        examples = [
            {
                "text": "I love the new features in this app!",
                "entity": "Twitter",
                "expected": "Positive"
            },
            {
                "text": "This service is terrible and slow.",
                "entity": "Amazon",
                "expected": "Negative"
            },
            {
                "text": "The company announced quarterly earnings today.",
                "entity": "Microsoft",
                "expected": "Neutral"
            }
        ]
        
        for i, example in enumerate(examples):
            st.markdown(f"**Example {i+1}:**")
            st.code(f"Text: {example['text']}\nEntity: {example['entity']}\nExpected: {example['expected']}")
            
            if st.button(f"Try Example {i+1}", key=f"example_{i}"):
                st.session_state.tweet_text = example["text"]
                st.session_state.entity = example["entity"]
                st.rerun()
    
    # Info section
    st.sidebar.markdown("---")
    st.sidebar.header("ℹ️ About")
    st.sidebar.info(
        "This app uses trained machine learning models to classify "
        "the sentiment of tweets. You can choose between a baseline "
        "TF-IDF model or a deep RNN model."
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Model Info")
    if model_type == "baseline":
        st.sidebar.info(
            "**Baseline Model:**\n"
            "- TF-IDF vectorization\n"
            "- LinearSVC classifier\n"
            "- Fast inference"
        )
    else:
        st.sidebar.info(
            "**RNN Model:**\n"
            "- Embedding layer\n"
            "- Bidirectional LSTM/GRU\n"
            "- Deep learning approach"
        )

else:
    st.error("Model not loaded. Please train the models first.")
    st.info("""
    To train the models, run:
    ```bash
    python -m src.train_baseline
    python -m src.train_rnn
    ```
    """)
