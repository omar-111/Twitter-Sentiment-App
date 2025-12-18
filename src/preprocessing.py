"""
Preprocessing utilities for Twitter text data.
Handles URL replacement, mentions, hashtags, and text normalization.
"""
import re
import pandas as pd
import numpy as np
import logging
from src.config import INPUT_MODE, MIN_TEXT_LENGTH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_tweet(text, reduce_repeated_chars=True):
    """
    Clean and normalize a single tweet text.
    
    Args:
        text: Raw tweet text (string)
        reduce_repeated_chars: Whether to reduce repeated characters (e.g., "goooood" -> "good")
        
    Returns:
        str: Cleaned tweet text
    """
    if pd.isna(text) or text == "":
        return ""
    
    text = str(text)
    
    # Replace URLs with <url> token
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_pattern, '<url>', text)
    
    # Replace mentions (@user) with <user> token
    text = re.sub(r'@\w+', '<user>', text)
    
    # Handle hashtags: remove '#' but keep the word
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Reduce repeated characters (optional)
    if reduce_repeated_chars:
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)  # "goooood" -> "good"
    
    # Remove weird symbols but keep basic punctuation
    # Keep: letters, numbers, spaces, and basic punctuation (. , ! ? : ; - ' " ( ) [ ])
    text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\'\"\(\)\[\]]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Lowercase
    text = text.lower()
    
    return text


def preprocess_dataframe(df):
    """
    Preprocess entire dataframe: clean text, handle NaNs, remove duplicates and short texts.
    
    Args:
        df: DataFrame with columns ["id", "entity", "sentiment", "text"]
        
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    logger.info("Starting preprocessing...")
    initial_len = len(df)
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Handle NaNs
    logger.info(f"Removing rows with NaN values...")
    df = df.dropna(subset=["text", "sentiment"])
    
    # Clean text column
    logger.info("Cleaning text column...")
    df["text"] = df["text"].apply(clean_tweet)
    
    # Remove empty or very short texts
    logger.info(f"Removing texts shorter than {MIN_TEXT_LENGTH} characters...")
    df = df[df["text"].str.len() >= MIN_TEXT_LENGTH]
    
    # Remove duplicates
    logger.info("Removing duplicate rows...")
    df = df.drop_duplicates(subset=["text", "sentiment"], keep="first")
    
    # Reset index
    df = df.reset_index(drop=True)
    
    final_len = len(df)
    logger.info(f"Preprocessing complete: {initial_len} -> {final_len} samples (removed {initial_len - final_len})")
    
    return df


def combine_entity_text(df):
    """
    Combine entity and text columns based on INPUT_MODE config.
    
    Args:
        df: DataFrame with columns ["id", "entity", "sentiment", "text"]
        
    Returns:
        pd.Series: Combined text series
    """
    if INPUT_MODE == "entity_text":
        # Combine entity and text
        combined = df["entity"].astype(str) + " " + df["text"].astype(str)
        return combined
    else:
        # Return text only
        return df["text"]


def prepare_features(df):
    """
    Prepare features for modeling based on INPUT_MODE.
    
    Args:
        df: Preprocessed DataFrame
        
    Returns:
        pd.Series: Feature text series ready for modeling
    """
    return combine_entity_text(df)
