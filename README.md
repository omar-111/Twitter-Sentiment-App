# Twitter Sentiment Classification Project

A complete, production-ready Twitter sentiment classification system that classifies tweets by sentiment (Positive, Negative, Neutral, Irrelevant) with entity-based context. The project includes comprehensive preprocessing, exploratory data analysis, two different model architectures, and a Streamlit web interface.

## 📁 Project Structure

```
.
├── src/
│   ├── config.py              # Configuration (paths, hyperparameters)
│   ├── data_io.py              # Data loading and artifact saving
│   ├── preprocessing.py       # Twitter text preprocessing
│   ├── eda.py                  # Exploratory Data Analysis
│   ├── train_baseline.py      # Baseline model training (TF-IDF + Linear)
│   ├── train_rnn.py           # RNN model training (Embedding + BiLSTM/GRU)
│   ├── evaluate.py            # Model evaluation and metrics
│   └── inference.py            # Inference utilities
├── apps/
│   └── streamlit_app.py       # Streamlit web application
├── models/                     # Saved model artifacts (created after training)
├── reports/                    # EDA plots and reports (created after EDA)
├── TWITTER_ZIP/               # Data directory
│   ├── twitter_training.csv
│   └── twitter_test.csv
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Exploratory Data Analysis

```bash
python -m src.eda
```

This will:
- Generate EDA plots in `reports/` directory
- Print key insights to console
- Save metrics summary to `reports/eda_insights.json`

### 3. Train Baseline Model

```bash
python -m src.train_baseline
```

This will:
- Train a TF-IDF + LinearSVC model
- Save model to `models/baseline.joblib`
- Save label encoder to `models/labels.pkl`

### 4. Train RNN Model

```bash
python -m src.train_rnn
```

This will:
- Train an Embedding + BiLSTM/GRU model
- Save model to `models/rnn.keras`
- Save tokenizer to `models/tokenizer.pkl`
- Save label encoder to `models/labels.pkl`

### 5. Evaluate Models

```bash
python -m src.evaluate
```

This will:
- Evaluate both models on the test set
- Generate confusion matrices
- Perform error analysis
- Print comparison metrics

### 6. Launch Streamlit App

```bash
streamlit run apps/streamlit_app.py
```

The app will open in your browser where you can:
- Enter tweet text
- Optionally specify entity
- Choose between baseline or RNN model
- View predictions with confidence scores

## 📊 Features

### Preprocessing
- URL replacement with `<url>` token
- Mention replacement with `<user>` token
- Hashtag normalization (remove `#`, keep word)
- Text normalization and cleaning
- Optional repeated character reduction
- Configurable input mode: `entity_text` or `text_only`

### Models

**Baseline Model:**
- TF-IDF vectorization (1-2 grams)
- LinearSVC classifier
- Fast inference
- Good baseline performance

**RNN Model:**
- Tokenizer with vocabulary size 10,000
- Embedding layer (128 dimensions)
- Bidirectional LSTM or GRU
- Dropout for regularization
- Early stopping and learning rate reduction
- Class weights for imbalanced data

### Evaluation Metrics
- Accuracy
- Macro F1 Score
- Micro F1 Score
- Confusion Matrix
- Classification Report
- Error Analysis

## ⚙️ Configuration

Edit `src/config.py` to customize:

- **INPUT_MODE**: `"entity_text"` (default) or `"text_only"`
- **BASELINE_MODEL_TYPE**: `"LinearSVC"` or `"LogisticRegression"`
- **RNN_MODEL_TYPE**: `"BiLSTM"` or `"GRU"`
- **Hyperparameters**: embedding dimensions, RNN units, dropout rates, etc.

## 📈 EDA Outputs

After running `python -m src.eda`, you'll find in `reports/`:

- `sentiment_distribution.png` - Class distribution
- `entity_distribution.png` - Entity counts and sentiment per entity
- `text_length_distribution.png` - Text length statistics
- `top_tokens.png` - Top unigrams and bigrams per sentiment
- `eda_insights.json` - Summary metrics

## 🎯 Usage Examples

### Using Inference API

```python
from src.inference import load_predictor

# Load predictor
predictor = load_predictor(model_type="baseline")  # or "rnn"

# Predict single text
result = predictor.predict(
    "I love this product!",
    entity="Twitter",
    return_proba=True
)

print(f"Label: {result['label']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Probabilities: {result['probabilities']}")
```

### Batch Prediction

```python
texts = ["Great service!", "Terrible experience."]
entities = ["Amazon", "Amazon"]
results = predictor.predict_batch(texts, entities, return_proba=True)
```

## 📝 Data Format

The CSV files should have no header and 4 columns:
- Column 0: ID
- Column 1: Entity
- Column 2: Sentiment (Positive, Negative, Neutral, Irrelevant)
- Column 3: Text

Example:
```
2401,Borderlands,Positive,"im getting on borderlands and i will murder you all ,"
```

## 🔧 Troubleshooting

**Model not found error:**
- Make sure you've trained the models first using `train_baseline.py` and `train_rnn.py`

**Memory errors:**
- Reduce `VOCAB_SIZE` or `TFIDF_MAX_FEATURES` in `config.py`
- Reduce `BATCH_SIZE` for RNN training

**Import errors:**
- Ensure you're running commands from the project root directory
- Verify all dependencies are installed: `pip install -r requirements.txt`

## 📄 License

This project is provided as-is for educational and research purposes.

## 👤 Author

Senior NLP Engineer

---

**Note:** Make sure to run the scripts in order: EDA → Train Baseline → Train RNN → Evaluate → Streamlit App
