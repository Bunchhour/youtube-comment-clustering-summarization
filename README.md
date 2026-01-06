# YouTube Comment Analyzer with AI Clustering

This application allows you to scrape comments from any YouTube video, analyze them using Machine Learning, and generate AI-powered summaries of public opinion.

It uses **Selenium/YouTube API** for scraping, **Sentence Transformers** for embedding, **K-Means** for clustering, and **Google Gemini AI** for summarizing each opinion cluster.

## ✨ Features

- **📥 Scrape Comments**: Fetch top-level comments from any YouTube video URL.
- **🔍 Intelligent Filtering**: Automatically filters out short, low-quality comments.
- **🧠 AI Clustering**: Groups similar comments into opinion clusters using semantic embeddings.
- **🤖 AI Summarization**: Uses Google Gemini to generate human-readable summaries for each cluster (Theme, Summary, Keywords).
- **📊 Interactive Dashboard**: Built with Streamlit for a smooth, user-friendly experience.
- **💾 Export Data**: Download the analyzed and clustered data as a CSV file.

## 🛠️ Prerequisites

- Python 3.8 or higher
- A [Google Cloud Project](https://console.cloud.google.com/) with **YouTube Data API v3** enabled.
- A [Google AI API Key](https://aistudio.google.com/) for Gemini (Gemini 2.5 Flash).

## 🚀 Installation

1.  **Clone the repository** (or download the files):
    ```bash
    git clone <repository-url>
    cd <project-folder>
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up Environment Variables**:
    Create a `.env` file in the root directory and add your API keys:
    ```ini
    YouTube_API=your_youtube_api_key_here
    GEMINI_API_KEY=your_gemini_api_key_here
    ```

## ▶️ Usage

1.  **Run the Streamlit App**:
    ```bash
    streamlit run streamlit_app.py
    ```

2.  **Analyze a Video**:
    - Enter a YouTube Video URL.
    - Click **"🚀 Scrape Comments"**.
    - Once scraped, adjust the number of clusters (default is 4).
    - Click **"🔬 Analyze & Cluster Comments"**.
    - View the AI-generated summaries and cluster distribution.

## 📂 Project Structure

- `streamlit_app.py`: Main application dashboard.
- `comment_scraping.py`: Handles YouTube API interactions and comment fetching.
- `machinelearning_pipeline.py`: Contains logic for text cleaning, embedding generation, and K-Means clustering.
- `ai_summarization.py`: Interface with Google Gemini for summarizing clustered comments.
- `requirements.txt`: List of Python dependencies.

## 🤖 Technologies Used

- **Frontend**: Streamlit
- **Scraping**: Google API Client (YouTube Data API v3)
- **ML/NLP**: Sentence Transformers (`all-MiniLM-L6-v2`), Scikit-learn (K-Means), NLTK, Emoji
- **AI GenAI**: Google Gemini 2.5 Flash
- **Data Handling**: Pandas, NumPy
