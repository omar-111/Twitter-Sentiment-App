"""
Train baseline model: TF-IDF + LinearSVC/LogisticRegression.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import joblib
import logging
from src.config import (
    RANDOM_SEED, BASELINE_MODEL_TYPE, TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE, BASELINE_C, TEST_SIZE, STRATIFY,
    BASELINE_MODEL_PATH, LABELS_PATH
)
from src.data_io import load_training_data, save_artifact
from src.preprocessing import preprocess_dataframe, prepare_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_baseline():
    """Train baseline model and save artifacts."""
    logger.info("="*60)
    logger.info("TRAINING BASELINE MODEL")
    logger.info("="*60)
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    df = load_training_data()
    df = preprocess_dataframe(df)
    
    # Prepare features and labels
    X = prepare_features(df)
    y = df["sentiment"]
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    logger.info(f"Training samples: {len(X)}")
    logger.info(f"Classes: {label_encoder.classes_}")
    logger.info(f"Class distribution:\n{y.value_counts()}")
    
    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_encoded if STRATIFY else None
    )
    
    logger.info(f"Train set: {len(X_train)}, Validation set: {len(X_val)}")
    
    # Create pipeline
    if BASELINE_MODEL_TYPE == "LinearSVC":
        model = LinearSVC(C=BASELINE_C, random_state=RANDOM_SEED, max_iter=2000)
    elif BASELINE_MODEL_TYPE == "LogisticRegression":
        model = LogisticRegression(C=BASELINE_C, random_state=RANDOM_SEED, max_iter=1000)
    else:
        raise ValueError(f"Unknown model type: {BASELINE_MODEL_TYPE}")
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            stop_words='english'
        )),
        ('model', model)
    ])
    
    # Train
    logger.info("Training baseline model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate on validation set
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    
    y_pred = pipeline.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average='macro')
    
    logger.info(f"Validation Accuracy: {accuracy:.4f}")
    logger.info(f"Validation Macro F1: {macro_f1:.4f}")
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_val, y_pred, target_names=label_encoder.classes_))
    
    # Save artifacts
    logger.info("Saving artifacts...")
    save_artifact(pipeline, BASELINE_MODEL_PATH)
    save_artifact(label_encoder, LABELS_PATH)
    
    logger.info("Baseline model training complete!")
    logger.info(f"Model saved to: {BASELINE_MODEL_PATH}")
    logger.info(f"Label encoder saved to: {LABELS_PATH}")
    
    return pipeline, label_encoder


if __name__ == "__main__":
    train_baseline()
