"""
Inference utilities for loading models and making predictions.
"""
import numpy as np
import pandas as pd
import logging
import tensorflow as tf
from src.config import (
    BASELINE_MODEL_PATH, RNN_MODEL_PATH, TOKENIZER_PATH, LABELS_PATH
)
from src.data_io import load_artifact
from src.preprocessing import clean_tweet
from src.config import INPUT_MODE, MAX_SEQUENCE_LENGTH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentPredictor:
    """Wrapper class for making sentiment predictions with either model."""
    
    def __init__(self, model_type="baseline"):
        """
        Initialize predictor with specified model type.
        
        Args:
            model_type: "baseline" or "rnn"
        """
        self.model_type = model_type
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load model and artifacts."""
        logger.info(f"Loading {self.model_type} model artifacts...")
        
        try:
            if self.model_type == "baseline":
                self.model = load_artifact(BASELINE_MODEL_PATH)
            elif self.model_type == "rnn":
                self.model = tf.keras.models.load_model("models/rnn.keras", compile=False)
                self.tokenizer = load_artifact(TOKENIZER_PATH)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            self.label_encoder = load_artifact(LABELS_PATH)
            logger.info("Artifacts loaded successfully")
            
        except FileNotFoundError as e:
            logger.error(f"Model artifacts not found: {e}")
            logger.error("Please train the models first using train_baseline.py or train_rnn.py")
            raise
    
    def preprocess_text(self, text, entity=None):
        """
        Preprocess input text for prediction.
        
        Args:
            text: Raw tweet text
            entity: Optional entity name
            
        Returns:
            str: Preprocessed text ready for model
        """
        # Clean text
        cleaned_text = clean_tweet(text)
        
        # Combine with entity if needed
        if INPUT_MODE == "entity_text" and entity:
            combined = str(entity).lower() + " " + cleaned_text
        else:
            combined = cleaned_text
        
        return combined
    
    def predict(self, text, entity=None, return_proba=False):
        """
        Predict sentiment for given text.
        
        Args:
            text: Raw tweet text
            entity: Optional entity name
            return_proba: If True, return probabilities for all classes
            
        Returns:
            dict: Prediction results with label, confidence, and optionally probabilities
        """
        # Preprocess
        processed_text = self.preprocess_text(text, entity)
        
        if self.model_type == "baseline":
            # Baseline model prediction
            if return_proba:
                # For LinearSVC, use decision_function and convert to probabilities
                decision_scores = self.model.decision_function([processed_text])[0]
                # Convert to probabilities using softmax
                exp_scores = np.exp(decision_scores - np.max(decision_scores))
                probabilities = exp_scores / exp_scores.sum()
            else:
                probabilities = None
            
            prediction = self.model.predict([processed_text])[0]
            label = self.label_encoder.inverse_transform([prediction])[0]
            
            if return_proba:
                confidence = float(np.max(probabilities))
            else:
                # Get confidence from decision function
                decision_scores = self.model.decision_function([processed_text])[0]
                confidence = float(np.max(decision_scores) - np.min(decision_scores)) / 2.0  # Normalized confidence
            
        elif self.model_type == "rnn":
            # RNN model prediction
            sequence = self.tokenizer.texts_to_sequences([processed_text])
            padded = tf.keras.preprocessing.sequence.pad_sequences(
                sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post'
            )
            
            probabilities = self.model.predict(padded, verbose=0)[0]
            prediction = np.argmax(probabilities)
            label = self.label_encoder.inverse_transform([prediction])[0]
            confidence = float(probabilities[prediction])
        
        result = {
            "label": label,
            "confidence": confidence
        }
        
        if return_proba:
            # Create probability dictionary
            proba_dict = {}
            for i, class_name in enumerate(self.label_encoder.classes_):
                proba_dict[class_name] = float(probabilities[i])
            result["probabilities"] = proba_dict
        
        return result
    
    def predict_batch(self, texts, entities=None, return_proba=False):
        """
        Predict sentiment for multiple texts.
        
        Args:
            texts: List of raw tweet texts
            entities: Optional list of entity names
            return_proba: If True, return probabilities for all classes
            
        Returns:
            list: List of prediction results
        """
        if entities is None:
            entities = [None] * len(texts)
        
        results = []
        for text, entity in zip(texts, entities):
            results.append(self.predict(text, entity, return_proba))
        
        return results


def load_predictor(model_type="baseline"):
    """
    Convenience function to load a predictor.
    
    Args:
        model_type: "baseline" or "rnn"
        
    Returns:
        SentimentPredictor: Loaded predictor instance
    """
    return SentimentPredictor(model_type=model_type)
