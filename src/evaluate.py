"""
Evaluation utilities: metrics, confusion matrix, and error analysis.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, precision_recall_fscore_support
)
import logging
from pathlib import Path
from src.config import REPORTS_DIR, BASELINE_MODEL_PATH, RNN_MODEL_PATH, TOKENIZER_PATH, LABELS_PATH
from src.data_io import load_test_data, load_artifact
from src.preprocessing import preprocess_dataframe, prepare_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")


def evaluate_baseline():
    """Evaluate baseline model on test set."""
    logger.info("="*60)
    logger.info("EVALUATING BASELINE MODEL")
    logger.info("="*60)
    
    # Load model and artifacts
    logger.info("Loading baseline model...")
    pipeline = load_artifact(BASELINE_MODEL_PATH)
    label_encoder = load_artifact(LABELS_PATH)
    
    # Load and preprocess test data
    logger.info("Loading test data...")
    df_test = load_test_data()
    df_test = preprocess_dataframe(df_test)
    
    X_test = prepare_features(df_test)
    y_test = df_test["sentiment"]
    y_test_encoded = label_encoder.transform(y_test)
    
    logger.info(f"Test samples: {len(X_test)}")
    
    # Predict
    logger.info("Making predictions...")
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.decision_function(X_test)  # For LinearSVC
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_encoded, y_pred)
    macro_f1 = f1_score(y_test_encoded, y_pred, average='macro')
    micro_f1 = f1_score(y_test_encoded, y_pred, average='micro')
    
    logger.info(f"\nTest Results:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Macro F1: {macro_f1:.4f}")
    logger.info(f"Micro F1: {micro_f1:.4f}")
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_))
    
    # Confusion matrix
    cm = confusion_matrix(y_test_encoded, y_pred)
    plot_confusion_matrix(cm, label_encoder.classes_, REPORTS_DIR / "baseline_confusion_matrix.png", "Baseline Model")
    
    # Error analysis
    error_analysis(df_test, y_test_encoded, y_pred, label_encoder, "baseline")
    
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "confusion_matrix": cm.tolist()
    }


def evaluate_rnn():
    """Evaluate RNN model on test set."""
    logger.info("="*60)
    logger.info("EVALUATING RNN MODEL")
    logger.info("="*60)
    
    # Load model and artifacts
    logger.info("Loading RNN model...")
    model = tf.keras.models.load_model("models/rnn.keras", compile=False)
    tokenizer = load_artifact(TOKENIZER_PATH)
    label_encoder = load_artifact(LABELS_PATH)
    
    # Load and preprocess test data
    logger.info("Loading test data...")
    df_test = load_test_data()
    df_test = preprocess_dataframe(df_test)
    
    X_test = prepare_features(df_test)
    y_test = df_test["sentiment"]
    y_test_encoded = label_encoder.transform(y_test)
    
    logger.info(f"Test samples: {len(X_test)}")
    
    # Tokenize and pad
    from src.config import MAX_SEQUENCE_LENGTH
    
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_test_padded = tf.keras.preprocessing.sequence.pad_sequences(
        X_test_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post'
    )
    
    # Predict
    logger.info("Making predictions...")
    y_pred_proba = model.predict(X_test_padded, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_encoded, y_pred)
    macro_f1 = f1_score(y_test_encoded, y_pred, average='macro')
    micro_f1 = f1_score(y_test_encoded, y_pred, average='micro')
    
    logger.info(f"\nTest Results:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Macro F1: {macro_f1:.4f}")
    logger.info(f"Micro F1: {micro_f1:.4f}")
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_))
    
    # Confusion matrix
    cm = confusion_matrix(y_test_encoded, y_pred)
    plot_confusion_matrix(cm, label_encoder.classes_, REPORTS_DIR / "rnn_confusion_matrix.png", "RNN Model")
    
    # Error analysis
    error_analysis(df_test, y_test_encoded, y_pred, label_encoder, "rnn")
    
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "confusion_matrix": cm.tolist()
    }


def plot_confusion_matrix(cm, class_names, save_path, title):
    """Plot and save confusion matrix."""
    logger.info(f"Plotting confusion matrix...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, cbar_kws={'label': 'Count'}
    )
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'Confusion Matrix - {title}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved confusion matrix to {save_path}")


def error_analysis(df_test, y_true, y_pred, label_encoder, model_name):
    """Perform error analysis and save examples."""
    logger.info(f"Performing error analysis for {model_name}...")
    
    # Get misclassified examples
    errors = df_test[y_true != y_pred].copy()
    errors["true_label"] = label_encoder.inverse_transform(y_true[y_true != y_pred])
    errors["predicted_label"] = label_encoder.inverse_transform(y_pred[y_true != y_pred])
    
    logger.info(f"Total misclassified: {len(errors)} ({len(errors)/len(df_test)*100:.2f}%)")
    
    # Save error examples
    error_path = REPORTS_DIR / f"{model_name}_error_examples.csv"
    errors[["entity", "sentiment", "text", "true_label", "predicted_label"]].to_csv(
        error_path, index=False
    )
    logger.info(f"Saved error examples to {error_path}")
    
    # Print some examples
    if len(errors) > 0:
        logger.info("\nSample error examples:")
        for idx, row in errors.head(5).iterrows():
            logger.info(f"\nText: {row['text'][:100]}...")
            logger.info(f"True: {row['true_label']}, Predicted: {row['predicted_label']}")


def run_evaluation():
    """Run evaluation for both models."""
    logger.info("Running evaluation on test set...")
    
    results = {}
    
    # Evaluate baseline
    try:
        results["baseline"] = evaluate_baseline()
    except Exception as e:
        logger.error(f"Error evaluating baseline: {e}")
    
    # Evaluate RNN
    try:
        results["rnn"] = evaluate_rnn()
    except Exception as e:
        logger.error(f"Error evaluating RNN: {e}")
    
    # Compare results
    if "baseline" in results and "rnn" in results:
        logger.info("\n" + "="*60)
        logger.info("MODEL COMPARISON")
        logger.info("="*60)
        logger.info(f"Baseline - Accuracy: {results['baseline']['accuracy']:.4f}, Macro F1: {results['baseline']['macro_f1']:.4f}")
        logger.info(f"RNN      - Accuracy: {results['rnn']['accuracy']:.4f}, Macro F1: {results['rnn']['macro_f1']:.4f}")
    
    return results


if __name__ == "__main__":
    run_evaluation()
