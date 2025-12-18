"""
Train RNN model: Embedding + BiLSTM/GRU for sentiment classification.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import logging
from src.config import (
    RANDOM_SEED, RNN_MODEL_TYPE, EMBEDDING_DIM, RNN_UNITS,
    DROPOUT_RATE, RECURRENT_DROPOUT_RATE, MAX_SEQUENCE_LENGTH,
    VOCAB_SIZE, BATCH_SIZE, EPOCHS, VALIDATION_SPLIT,
    EARLY_STOPPING_PATIENCE, REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR,
    TEST_SIZE, STRATIFY, RNN_MODEL_PATH, TOKENIZER_PATH, LABELS_PATH
)
from src.data_io import load_training_data, save_artifact
from src.preprocessing import preprocess_dataframe, prepare_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
import tensorflow as tf
tf.random.set_seed(RANDOM_SEED)


def create_rnn_model(vocab_size, embedding_dim, max_length, num_classes, rnn_units, dropout_rate, recurrent_dropout_rate):
    """
    Create RNN model architecture.
    
    Args:
        vocab_size: Vocabulary size
        embedding_dim: Embedding dimension
        max_length: Maximum sequence length
        num_classes: Number of output classes
        rnn_units: Number of RNN units
        dropout_rate: Dropout rate
        recurrent_dropout_rate: Recurrent dropout rate
        
    Returns:
        keras.Model: Compiled model
    """
    model = keras.Sequential([
        layers.Embedding(vocab_size + 1, embedding_dim, input_length=max_length),
        layers.Dropout(dropout_rate),
    ])
    
    if RNN_MODEL_TYPE == "BiLSTM":
        model.add(layers.Bidirectional(
            layers.LSTM(rnn_units, dropout=dropout_rate, recurrent_dropout=recurrent_dropout_rate, return_sequences=True)
        ))
        model.add(layers.Bidirectional(
            layers.LSTM(rnn_units // 2, dropout=dropout_rate, recurrent_dropout=recurrent_dropout_rate)
        ))
    elif RNN_MODEL_TYPE == "GRU":
        model.add(layers.Bidirectional(
            layers.GRU(rnn_units, dropout=dropout_rate, recurrent_dropout=recurrent_dropout_rate, return_sequences=True)
        ))
        model.add(layers.Bidirectional(
            layers.GRU(rnn_units // 2, dropout=dropout_rate, recurrent_dropout=recurrent_dropout_rate)
        ))
    else:
        raise ValueError(f"Unknown RNN model type: {RNN_MODEL_TYPE}")
    
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_rnn():
    """Train RNN model and save artifacts."""
    logger.info("="*60)
    logger.info("TRAINING RNN MODEL")
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
    
    # Tokenize and pad sequences
    logger.info("Tokenizing and padding sequences...")
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_val_seq = tokenizer.texts_to_sequences(X_val)
    
    X_train_padded = pad_sequences(X_train_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    X_val_padded = pad_sequences(X_val_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    
    logger.info(f"Vocabulary size: {len(tokenizer.word_index)}")
    logger.info(f"Sequence length: {MAX_SEQUENCE_LENGTH}")
    
    # Calculate class weights for imbalanced data
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    logger.info(f"Class weights: {class_weight_dict}")
    
    # Create model
    num_classes = len(label_encoder.classes_)
    logger.info(f"Creating {RNN_MODEL_TYPE} model...")
    model = create_rnn_model(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        max_length=MAX_SEQUENCE_LENGTH,
        num_classes=num_classes,
        rnn_units=RNN_UNITS,
        dropout_rate=DROPOUT_RATE,
        recurrent_dropout_rate=RECURRENT_DROPOUT_RATE
    )
    
    logger.info("Model architecture:")
    model.summary(print_fn=logger.info)
    
    # Callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=REDUCE_LR_FACTOR,
        patience=REDUCE_LR_PATIENCE,
        min_lr=1e-7,
        verbose=1
    )
    
    # Train model
    logger.info("Training RNN model...")
    history = model.fit(
        X_train_padded, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val_padded, y_val),
        class_weight=class_weight_dict,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    val_loss, val_accuracy = model.evaluate(X_val_padded, y_val, verbose=0)
    logger.info(f"Validation Loss: {val_loss:.4f}")
    logger.info(f"Validation Accuracy: {val_accuracy:.4f}")
    
    # Calculate F1 score
    from sklearn.metrics import f1_score, classification_report
    y_pred_proba = model.predict(X_val_padded, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    macro_f1 = f1_score(y_val, y_pred, average='macro')
    logger.info(f"Validation Macro F1: {macro_f1:.4f}")
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_val, y_pred, target_names=label_encoder.classes_))
    
    # Save artifacts
    logger.info("Saving artifacts...")
    model.save(RNN_MODEL_PATH)
    save_artifact(tokenizer, TOKENIZER_PATH)
    save_artifact(label_encoder, LABELS_PATH)
    
    logger.info("RNN model training complete!")
    logger.info(f"Model saved to: {RNN_MODEL_PATH}")
    logger.info(f"Tokenizer saved to: {TOKENIZER_PATH}")
    logger.info(f"Label encoder saved to: {LABELS_PATH}")
    
    return model, tokenizer, label_encoder


if __name__ == "__main__":
    train_rnn()
