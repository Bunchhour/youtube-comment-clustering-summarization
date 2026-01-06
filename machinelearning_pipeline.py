"""
YouTube Comment Analysis Script
This script performs text cleaning, preprocessing, and clustering on YouTube comments.
"""
import os
import pandas as pd
import numpy as np
import re
import emoji
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from glob import glob
import streamlit as st

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

def get_latest_csv(folder="scraped_data"):
    """
    Get the most recently created CSV file from the scraped data folder.
    
    Args:
        folder: Path to the folder containing scraped CSV files
        
    Returns:
        Path to the latest CSV file
    """
    # Get all CSV files in the folder
    csv_files = glob(os.path.join(folder, "youtube_comments_*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in '{folder}' folder")
    
    # Get the most recent file by modification time
    latest_file = max(csv_files, key=os.path.getmtime)
    return latest_file

def load_comments_data(filepath=None, folder="scraped_data"):
    """
    Load YouTube comments data from automatically saved CSV file.
    If no filepath is specified, automatically loads the most recently scraped file
    from the designated folder.
    
    Args:
        filepath: Specific file path to load. If None, auto-loads the latest scraped file.
        folder: Folder where scraped CSV files are stored (default: "scraped_data")
        
    Returns:
        pandas DataFrame with scraped YouTube comments data
    """
    if filepath is None:
        filepath = get_latest_csv(folder)
    
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} comments")
    return df

def load_and_prepare_dataset(file_path=None, folder="scraped_data"):
    """
    Load the YouTube comments dataset from automatically saved CSV and standardize column names.
    If no file path is provided, automatically loads the most recently scraped file.
    
    Args:
        file_path: Path to the CSV file. If None, auto-loads the latest file from scraped_data folder.
        folder: Folder where scraped CSV files are stored (default: "scraped_data")
        
    Returns:
        DataFrame with standardized column names (lowercase, no spaces, stripped whitespace)
    """
    # Auto-load latest file if no path provided
    if file_path is None:
        file_path = get_latest_csv(folder)
        print(f"Auto-loading latest file: {file_path}")
    
    dataset = pd.read_csv(file_path)
    
    # Standardize column names: strip whitespace, convert to lowercase, replace spaces with underscores
    dataset.columns = (
        dataset.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    
    print(f"Loaded {len(dataset)} comments with columns: {list(dataset.columns)}")
    
    return dataset


def has_min_words(text, min_words=3):
    """
    Check if text has at least the minimum number of words.
    
    Args:
        text: Input text string
        min_words: Minimum word count threshold (default: 3)
        
    Returns:
        Boolean indicating if text meets minimum word requirement
    """
    # Return False for empty or null text
    if pd.isna(text) or text.strip() == '':
        return False
    
    # Count words by splitting on whitespace
    word_count = len(text.split())
    return word_count >= min_words


def remove_urls(text):
    """
    Remove URLs from text.
    
    Args:
        text: Input text string
        
    Returns:
        Text with URLs removed
    """
    # Pattern matches http/https URLs and www URLs
    url_pattern = r'https?://\S+|www\.\S+'
    return re.sub(url_pattern, '', text)


def emoji_to_text(text):
    """
    Convert emojis to text representation.
    
    Args:
        text: Input text string
        
    Returns:
        Text with emojis converted to readable names
    """
    # Converts emojis to :emoji_name: format with space delimiters
    return emoji.demojize(text, delimiters=(" ", " "))


def remove_timestamps(text):
    """
    Remove timestamp patterns from text.
    
    Args:
        text: Input text string
        
    Returns:
        Text with timestamps removed
    """
    # Pattern matches timestamps like 1:23 or 12:34:56
    timestamp_pattern = r'\b\d{1,2}:\d{2}(:\d{2})?\b'
    return re.sub(timestamp_pattern, '', text)


def emoji_to_sentiment(text):
    """
    Convert specific emojis to sentiment tokens.
    Maps common emojis to standardized emotion tokens (EMO_SAD, EMO_LOVE, etc.)
    
    Args:
        text: Input text string
        
    Returns:
        Text with emojis replaced by sentiment tokens
    """
    # Mapping of emoji names to sentiment tokens
    EMOJI_SENTIMENT_MAP = {
        "crying_face": "EMO_SAD",
        "loudly_crying_face": "EMO_SAD",
        "broken_heart": "EMO_SAD",
        
        "red_heart": "EMO_LOVE",
        "heart": "EMO_LOVE",
        "smiling_face_with_heart_eyes": "EMO_LOVE",

        "grinning_face": "EMO_HAPPY",
        "smiling_face": "EMO_HAPPY",
        "face_with_tears_of_joy": "EMO_HAPPY",

        "angry_face": "EMO_ANGRY",
        "pouting_face": "EMO_ANGRY",
    }
    
    # First convert emojis to text representation
    text = emoji.demojize(text, delimiters=(" ", " "))
    
    # Replace emoji names with sentiment tokens
    for emoji_name, token in EMOJI_SENTIMENT_MAP.items():
        text = re.sub(rf"\b{emoji_name}\b", token, text)
    
    return text


def remove_stopwords(text):
    """
    Remove stopwords while preserving important sentiment words.
    Keeps negation words, intensifiers, and emotion tokens.
    
    Args:
        text: Input text string
        
    Returns:
        Text with stopwords removed
    """
    # Load standard English stopwords
    stop_words = set(stopwords.words('english'))
    
    # Words to keep despite being stopwords (important for sentiment)
    KEEP_WORDS = {
        "not", "no", "nor", "never",  # Negation words
        "very", "too", "so",  # Intensifiers
        "EMO_SAD", "EMO_LOVE", "EMO_HAPPY", "EMO_ANGRY"  # Emotion tokens
    }
    
    # Create custom stopwords set excluding important words
    custom_stopwords = stop_words - KEEP_WORDS
    
    # Remove stopwords from text
    words = text.split()
    return " ".join(
        word for word in words if word.lower() not in custom_stopwords
    )


def clean_baseline_comments(dataframe):
    """
    Apply baseline cleaning pipeline with stopword removal.
    
    Args:
        dataframe: DataFrame with 'comment_text' column
        
    Returns:
        DataFrame with 'comment_clean' column added
    """
    # Chain all cleaning operations
    dataframe['comment_clean'] = (
        dataframe['comment_text']
        .str.lower()  # Convert to lowercase
        .apply(remove_urls)  # Remove URLs
        .apply(emoji_to_sentiment)  # Convert emojis to sentiment tokens
        .apply(remove_timestamps)  # Remove timestamps
        .apply(remove_stopwords)  # Remove stopwords
        .str.strip()  # Remove leading/trailing whitespace
    )
    
    return dataframe


def clean_improved_comments(dataframe):
    """
    Apply improved cleaning pipeline without stopword removal.
    Preserves more context by keeping all words.
    
    Args:
        dataframe: DataFrame with 'comment_text' column
        
    Returns:
        DataFrame with 'comment_clean' column added
    """
    # Chain all cleaning operations (no stopword removal)
    dataframe['comment_clean'] = (
        dataframe['comment_text']
        .str.lower()  # Convert to lowercase
        .apply(remove_urls)  # Remove URLs
        .apply(emoji_to_sentiment)  # Convert emojis to sentiment tokens
        .apply(remove_timestamps)  # Remove timestamps
        .str.strip()  # Remove leading/trailing whitespace
    )
    
    return dataframe


@st.cache_resource
def load_model():
    """Load and cache the sentence transformer model."""
    return SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(comments_list, show_progress=False):
    """
    Generate sentence embeddings using pre-trained transformer model.
    Converts text into dense vector representations for similarity analysis.
    
    Args:
        comments_list: List of cleaned comment strings
        show_progress: Whether to show progress bar
        
    Returns:
        NumPy array of embeddings (shape: [n_comments, 384])
    """
    # Load pre-trained sentence transformer model
    model = load_model()
    
    # Generate embeddings for all comments
    embeddings = model.encode(
        comments_list,
        show_progress_bar=show_progress,  # Display progress during encoding
        convert_to_numpy=True  # Return as NumPy array
    )
    
    if show_progress:
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Each comment is now a {embeddings.shape[1]}-dimensional vector")
    
    return embeddings


def perform_clustering(embeddings, n_clusters=4):
    """
    Perform K-Means clustering on embeddings.
    Groups similar comments together based on semantic meaning.
    
    Args:
        embeddings: NumPy array of comment embeddings
        n_clusters: Number of clusters to create
        
    Returns:
        Array of cluster labels for each comment
    """
    # Initialize K-Means clustering
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,  # For reproducibility
        n_init=10,  # Number of initializations
        max_iter=300  # Maximum iterations
    )
    
    # Fit model and predict cluster labels
    clusters = kmeans.fit_predict(embeddings)
    
    return clusters


def display_cluster_results(dataframe, n_clusters):
    """
    Display cluster distribution and sample comments from each cluster.
    
    Args:
        dataframe: DataFrame with 'cluster' and 'comment_clean' columns
        n_clusters: Number of clusters
    """
    # Show cluster distribution
    print("\nCluster distribution (Improved Model):")
    print(dataframe['cluster'].value_counts().sort_index())
    
    # Show sample comments from each cluster
    for i in range(n_clusters):
        print(f"\n--- Cluster {i} (Sample Comments) ---")
        cluster_comments = dataframe[
            dataframe['cluster'] == i
        ]['comment_clean'].head(5)
        
        for idx, comment in enumerate(cluster_comments, 1):
            print(f"{idx}. {comment}")


def main():
    """
    Main execution function.
    Orchestrates the entire comment analysis pipeline.
    """
    # Step 1: Load and prepare dataset
    
    dataset = load_comments_data()
    
    # Step 2: Extract relevant column
    cleaning_dataset = dataset[['comment_text']].copy()
    
    print("Initial dataset loaded.")
    print(cleaning_dataset.head())
    
    # Step 3: Filter out comments with fewer than 3 words
    cleaning_dataset = cleaning_dataset[
        cleaning_dataset['comment_text'].apply(has_min_words)
    ]
    
    # Reset index after filtering
    cleaning_dataset = cleaning_dataset.reset_index(drop=True)
    
    print(f"\nDataset after filtering: {len(cleaning_dataset)} comments remaining")
    
    # Step 4: Create baseline and improved datasets
    cleaning_dataset_baseline = cleaning_dataset.copy()
    cleaning_dataset_improved = cleaning_dataset.copy()
    
    # Step 5: Apply cleaning pipelines
    cleaning_dataset_baseline = clean_baseline_comments(cleaning_dataset_baseline)
    cleaning_dataset_improved = clean_improved_comments(cleaning_dataset_improved)
    
    print("\nCleaning complete.")
    print(cleaning_dataset_improved[['comment_text', 'comment_clean']].head())
    
    # Step 6: Generate embeddings from improved dataset
    embeddings = generate_embeddings(
        cleaning_dataset_improved['comment_clean'].tolist(),
        show_progress=True
    )
    
    # Step 7: Perform clustering
    n_clusters_improved = 4
    clusters_improved = perform_clustering(embeddings, n_clusters_improved)
    
    # Step 8: Add cluster labels to dataset
    cleaning_dataset_improved['cluster'] = clusters_improved
    
    # Step 9: Display results
    display_cluster_results(cleaning_dataset_improved, n_clusters_improved)
    
    # Optional: Save results to CSV
    # cleaning_dataset_improved.to_csv('clustered_comments.csv', index=False)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()