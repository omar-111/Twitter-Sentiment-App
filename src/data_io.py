"""
Data I/O utilities for loading CSV files and saving/loading artifacts.
"""
import pandas as pd
import pickle
import joblib
from pathlib import Path
import logging
from src.config import TRAIN_CSV, TEST_CSV, MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_training_data():
    """
    Load training CSV file with no header.
    
    Returns:
        pd.DataFrame: DataFrame with columns ["id", "entity", "sentiment", "text"]
    """
    logger.info(f"Loading training data from {TRAIN_CSV}")
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"Training CSV not found at {TRAIN_CSV}")
    
    # Handle different pandas versions
    try:
        df = pd.read_csv(
            TRAIN_CSV,
            header=None,
            names=["id", "entity", "sentiment", "text"],
            encoding="utf-8",
            on_bad_lines="skip"
        )
    except TypeError:
        # Older pandas version
        df = pd.read_csv(
            TRAIN_CSV,
            header=None,
            names=["id", "entity", "sentiment", "text"],
            encoding="utf-8",
            error_bad_lines=False,
            warn_bad_lines=False
        )
    logger.info(f"Loaded {len(df)} training samples")
    return df


def load_test_data():
    """
    Load test CSV file with no header.
    
    Returns:
        pd.DataFrame: DataFrame with columns ["id", "entity", "sentiment", "text"]
    """
    logger.info(f"Loading test data from {TEST_CSV}")
    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Test CSV not found at {TEST_CSV}")
    
    # Handle different pandas versions
    try:
        df = pd.read_csv(
            TEST_CSV,
            header=None,
            names=["id", "entity", "sentiment", "text"],
            encoding="utf-8",
            on_bad_lines="skip"
        )
    except TypeError:
        # Older pandas version
        df = pd.read_csv(
            TEST_CSV,
            header=None,
            names=["id", "entity", "sentiment", "text"],
            encoding="utf-8",
            error_bad_lines=False,
            warn_bad_lines=False
        )
    logger.info(f"Loaded {len(df)} test samples")
    return df


def save_artifact(obj, filepath):
    """
    Save an artifact (model, tokenizer, etc.) to disk.
    
    Args:
        obj: Object to save (model, tokenizer, etc.)
        filepath: Path to save the object
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if filepath.suffix == ".joblib":
        joblib.dump(obj, filepath)
    elif filepath.suffix == ".pkl":
        with open(filepath, "wb") as f:
            pickle.dump(obj, f)
    else:
        raise ValueError(f"Unsupported file extension: {filepath.suffix}")
    
    logger.info(f"Saved artifact to {filepath}")


def load_artifact(filepath):
    """
    Load an artifact from disk.
    
    Args:
        filepath: Path to the artifact file
        
    Returns:
        Loaded object
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Artifact not found: {filepath}")
    
    if filepath.suffix == ".joblib":
        obj = joblib.load(filepath)
    elif filepath.suffix == ".pkl":
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
    else:
        raise ValueError(f"Unsupported file extension: {filepath.suffix}")
    
    logger.info(f"Loaded artifact from {filepath}")
    return obj
