"""
Exploratory Data Analysis (EDA) for Twitter sentiment data.
Generates plots and insights, saves to reports/ directory.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from src.config import REPORTS_DIR, RANDOM_SEED
from src.data_io import load_training_data
from src.preprocessing import preprocess_dataframe, prepare_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_sentiment_distribution(df, save_path):
    """Plot sentiment class distribution."""
    logger.info("Plotting sentiment distribution...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sentiment_counts = df["sentiment"].value_counts()
    colors = sns.color_palette("husl", len(sentiment_counts))
    
    bars = ax.bar(sentiment_counts.index, sentiment_counts.values, color=colors)
    ax.set_xlabel("Sentiment", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Sentiment Class Distribution", fontsize=14, fontweight="bold")
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved sentiment distribution plot to {save_path}")


def plot_entity_distribution(df, save_path):
    """Plot entity distribution and sentiment per entity."""
    logger.info("Plotting entity distribution...")
    
    # Top entities
    top_entities = df["entity"].value_counts().head(15)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Entity counts
    ax1.barh(range(len(top_entities)), top_entities.values, color=sns.color_palette("viridis", len(top_entities)))
    ax1.set_yticks(range(len(top_entities)))
    ax1.set_yticklabels(top_entities.index)
    ax1.set_xlabel("Count", fontsize=12)
    ax1.set_title("Top 15 Entities by Count", fontsize=14, fontweight="bold")
    ax1.invert_yaxis()
    
    # Sentiment per entity (for top 5 entities)
    top_5_entities = top_entities.head(5).index
    entity_sentiment = df[df["entity"].isin(top_5_entities)].groupby(["entity", "sentiment"]).size().unstack(fill_value=0)
    
    entity_sentiment.plot(kind="bar", stacked=True, ax=ax2, color=sns.color_palette("Set2", len(entity_sentiment.columns)))
    ax2.set_xlabel("Entity", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Sentiment Distribution for Top 5 Entities", fontsize=14, fontweight="bold")
    ax2.legend(title="Sentiment", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved entity distribution plot to {save_path}")


def plot_text_length_distribution(df, save_path):
    """Plot text length distribution."""
    logger.info("Plotting text length distribution...")
    
    # Prepare features to get actual text lengths
    features = prepare_features(df)
    text_lengths = features.str.len()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Histogram
    ax1.hist(text_lengths, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel("Text Length (characters)", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.set_title("Text Length Distribution (Histogram)", fontsize=14, fontweight="bold")
    ax1.axvline(text_lengths.mean(), color='red', linestyle='--', label=f'Mean: {text_lengths.mean():.1f}')
    ax1.axvline(text_lengths.median(), color='green', linestyle='--', label=f'Median: {text_lengths.median():.1f}')
    ax1.legend()
    
    # Box plot by sentiment
    df_with_length = df.copy()
    df_with_length["text_length"] = text_lengths
    sns.boxplot(data=df_with_length, x="sentiment", y="text_length", ax=ax2)
    ax2.set_xlabel("Sentiment", fontsize=12)
    ax2.set_ylabel("Text Length (characters)", fontsize=12)
    ax2.set_title("Text Length by Sentiment", fontsize=14, fontweight="bold")
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved text length distribution plot to {save_path}")


def plot_top_tokens(df, save_path, top_n=20):
    """Plot top tokens and bigrams per sentiment class using TF-IDF."""
    logger.info("Plotting top tokens per sentiment...")
    
    features = prepare_features(df)
    sentiments = df["sentiment"].unique()
    
    fig, axes = plt.subplots(len(sentiments), 2, figsize=(16, 5 * len(sentiments)))
    if len(sentiments) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, sentiment in enumerate(sentiments):
        # Filter data for this sentiment
        sentiment_data = features[df["sentiment"] == sentiment]
        
        # Unigrams
        vectorizer_uni = TfidfVectorizer(max_features=top_n, ngram_range=(1, 1), stop_words='english')
        tfidf_uni = vectorizer_uni.fit_transform(sentiment_data)
        feature_names_uni = vectorizer_uni.get_feature_names_out()
        scores_uni = tfidf_uni.sum(axis=0).A1
        top_indices_uni = scores_uni.argsort()[-top_n:][::-1]
        
        axes[idx, 0].barh(range(len(top_indices_uni)), scores_uni[top_indices_uni], color='steelblue')
        axes[idx, 0].set_yticks(range(len(top_indices_uni)))
        axes[idx, 0].set_yticklabels([feature_names_uni[i] for i in top_indices_uni])
        axes[idx, 0].set_xlabel("TF-IDF Score", fontsize=11)
        axes[idx, 0].set_title(f"Top {top_n} Unigrams - {sentiment}", fontsize=12, fontweight="bold")
        axes[idx, 0].invert_yaxis()
        
        # Bigrams
        vectorizer_bi = TfidfVectorizer(max_features=top_n, ngram_range=(2, 2), stop_words='english')
        tfidf_bi = vectorizer_bi.fit_transform(sentiment_data)
        feature_names_bi = vectorizer_bi.get_feature_names_out()
        scores_bi = tfidf_bi.sum(axis=0).A1
        top_indices_bi = scores_bi.argsort()[-top_n:][::-1]
        
        axes[idx, 1].barh(range(len(top_indices_bi)), scores_bi[top_indices_bi], color='coral')
        axes[idx, 1].set_yticks(range(len(top_indices_bi)))
        axes[idx, 1].set_yticklabels([feature_names_bi[i] for i in top_indices_bi])
        axes[idx, 1].set_xlabel("TF-IDF Score", fontsize=11)
        axes[idx, 1].set_title(f"Top {top_n} Bigrams - {sentiment}", fontsize=12, fontweight="bold")
        axes[idx, 1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved top tokens plot to {save_path}")


def generate_insights(df):
    """Generate and print key insights."""
    logger.info("Generating insights...")
    
    features = prepare_features(df)
    text_lengths = features.str.len()
    
    insights = {
        "total_samples": len(df),
        "sentiment_distribution": df["sentiment"].value_counts().to_dict(),
        "num_entities": df["entity"].nunique(),
        "top_entities": df["entity"].value_counts().head(10).to_dict(),
        "text_length_stats": {
            "mean": float(text_lengths.mean()),
            "median": float(text_lengths.median()),
            "std": float(text_lengths.std()),
            "min": int(text_lengths.min()),
            "max": int(text_lengths.max())
        },
        "class_imbalance_ratio": float(df["sentiment"].value_counts().max() / df["sentiment"].value_counts().min())
    }
    
    # Print insights
    print("\n" + "="*60)
    print("KEY INSIGHTS SUMMARY")
    print("="*60)
    print(f"1. Total Samples: {insights['total_samples']:,}")
    print(f"2. Number of Unique Entities: {insights['num_entities']}")
    print(f"3. Sentiment Distribution:")
    for sentiment, count in insights['sentiment_distribution'].items():
        pct = (count / insights['total_samples']) * 100
        print(f"   - {sentiment}: {count:,} ({pct:.1f}%)")
    print(f"4. Text Length Statistics:")
    print(f"   - Mean: {insights['text_length_stats']['mean']:.1f} characters")
    print(f"   - Median: {insights['text_length_stats']['median']:.1f} characters")
    print(f"   - Range: {insights['text_length_stats']['min']} - {insights['text_length_stats']['max']} characters")
    print(f"5. Class Imbalance Ratio: {insights['class_imbalance_ratio']:.2f}x")
    print("="*60 + "\n")
    
    return insights


def run_eda():
    """Main EDA function that generates all plots and insights."""
    logger.info("Starting EDA...")
    
    # Load and preprocess data
    df = load_training_data()
    df = preprocess_dataframe(df)
    
    # Generate plots
    plot_sentiment_distribution(df, REPORTS_DIR / "sentiment_distribution.png")
    plot_entity_distribution(df, REPORTS_DIR / "entity_distribution.png")
    plot_text_length_distribution(df, REPORTS_DIR / "text_length_distribution.png")
    plot_top_tokens(df, REPORTS_DIR / "top_tokens.png")
    
    # Generate insights
    insights = generate_insights(df)
    
    # Save insights to JSON
    insights_path = REPORTS_DIR / "eda_insights.json"
    with open(insights_path, "w") as f:
        json.dump(insights, f, indent=2)
    logger.info(f"Saved insights to {insights_path}")
    
    logger.info("EDA complete!")
    return df, insights


if __name__ == "__main__":
    run_eda()
