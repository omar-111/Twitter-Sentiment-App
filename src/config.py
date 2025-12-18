"""
Configuration file for Twitter Sentiment Classification Project.
Contains paths, seeds, and hyperparameters.
"""
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "TWITTER_ZIP"
TRAIN_CSV = DATA_DIR / "twitter_training.csv"
TEST_CSV = DATA_DIR / "twitter_test.csv"

# Fallback to root if TWITTER_ZIP doesn't exist
if not TRAIN_CSV.exists():
    TRAIN_CSV = PROJECT_ROOT / "twitter_training.csv"
if not TEST_CSV.exists():
    TEST_CSV = PROJECT_ROOT / "twitter_test.csv"

# Output directories
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# Reproducibility
RANDOM_SEED = 42

# Preprocessing config
INPUT_MODE = "entity_text"  # Options: "entity_text" or "text_only"
MIN_TEXT_LENGTH = 3  # Minimum character length for valid text

# Baseline model config
BASELINE_MODEL_TYPE = "LinearSVC"  # Options: "LinearSVC" or "LogisticRegression"
TFIDF_MAX_FEATURES = 10000
TFIDF_NGRAM_RANGE = (1, 2)
BASELINE_C = 1.0  # Regularization parameter

# RNN model config
RNN_MODEL_TYPE = "BiLSTM"  # Options: "BiLSTM" or "GRU"
EMBEDDING_DIM = 128
RNN_UNITS = 128
DROPOUT_RATE = 0.3
RECURRENT_DROPOUT_RATE = 0.2
MAX_SEQUENCE_LENGTH = 200
VOCAB_SIZE = 10000
BATCH_SIZE = 64
EPOCHS = 20
VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5

# Training config
TEST_SIZE = 0.2
STRATIFY = True

# Artifact filenames
BASELINE_MODEL_PATH = MODELS_DIR / "baseline.joblib"
RNN_MODEL_PATH = MODELS_DIR / "rnn.keras"
TOKENIZER_PATH = MODELS_DIR / "tokenizer.pkl"
LABELS_PATH = MODELS_DIR / "labels.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"  # For baseline if needed
