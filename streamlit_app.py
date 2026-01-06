import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from datetime import datetime
from io import BytesIO

# Import AI summarization
from ai_summarization import summarize_comment_cluster

# Import comment scraping functions
from comment_scraping import (
    get_video_id,
    get_video_metadata,
)

# Import ML pipeline functions (THIS WAS MISSING!)
from machinelearning_pipeline import (
    clean_improved_comments,
    generate_embeddings,
    perform_clustering,
    has_min_words
)

def scrape_with_progress(api_key, video_url, include_replies, progress_bar, status_text):
    """Wrapper to add Streamlit progress tracking to the scrape function."""
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    import time
    
    video_id = get_video_id(video_url)
    if not video_id:
        st.error("Invalid YouTube URL.")
        return None, None

    youtube = build('youtube', 'v3', developerKey=api_key)
    
    if status_text:
        status_text.text("Fetching video metadata...")
    
    video_metadata = get_video_metadata(youtube, video_id)
    
    if not video_metadata:
        return None, None

    comments_data = []
    next_page_token = None
    page_count = 0

    try:
        while True:
            request = youtube.commentThreads().list(
                part="snippet,replies" if include_replies else "snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token,
                textFormat="plainText"
            )
            response = request.execute()
            page_count += 1

            for item in response.get('items', []):
                top_comment = item['snippet']['topLevelComment']['snippet']
                
                comments_data.append({
                    'comment_id': item['snippet']['topLevelComment']['id'],
                    'comment_text': top_comment['textOriginal'],
                    'author': top_comment['authorDisplayName'],
                    'like_count': top_comment['likeCount'],
                    'published_at': top_comment['publishedAt'],
                    'is_reply': False
                })
                
            if status_text:
                status_text.text(f"Page {page_count}: Collected {len(comments_data)} comments/replies...")
            
            if progress_bar and video_metadata['comment_count'] != 'N/A':
                try:
                    total_comments = int(video_metadata['comment_count'])
                    progress = min(len(comments_data) / total_comments, 1.0)
                    progress_bar.progress(progress)
                except:
                    pass
            
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
            
            time.sleep(0.5)

        df = pd.DataFrame(comments_data)
        return df, video_metadata

    except HttpError as e:
        error_content = e.content.decode('utf-8')
        if 'commentsDisabled' in error_content:
            st.error("Comments are disabled for this video.")
        elif 'quotaExceeded' in error_content:
            st.error("YouTube API quota exceeded. Try again tomorrow or use a different API key.")
        elif 'videoNotFound' in error_content:
            st.error("Video not found or is private/deleted.")
        else:
            st.error(f"API Error: {e}")
        return None, None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None, None

def auto_save_to_folder(df, video_id, save_folder="scraped_data"):
    """Automatically save DataFrame to a designated folder and return the file path."""
    # Create folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"youtube_comments_{video_id}_{timestamp}.csv"
    filepath = os.path.join(save_folder, filename)
    
    # Save the file
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    
    return filepath


def main():
    # Load environment variables
    load_dotenv()
    
    # Page config
    # st.set_page_config(
    #     page_title="YouTube Comment Analyzer",
    #     page_icon="🎥",
    #     layout="wide"
    # )
    
    # Logo (if exists)
    if os.path.exists("image/Logo.png"):
        st.image("image/Logo.png")
    
    # Header image
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if os.path.exists("image/YouTube_comment.png"):
            st.image("image/YouTube_comment.png", width=400)
    
    st.title("Welcome to YouTube Comment Analyzer with AI Clustering")
    st.markdown("Scrape comments, cluster by topic, and get AI-powered summaries!")

    video_url = st.text_input(
        "YouTube Video URL",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Paste a YouTube video link here"
    )

    youtube_api_key = os.getenv("YouTube_API")
    save_folder = "scraped_data"

    # === PART 1: SCRAPE COMMENTS ===
    st.markdown("---")
    st.header("📥 Step 1: Scrape Comments")
    
    if st.button("🚀 Scrape Comments", type="primary", use_container_width=True):
        if not video_url:
            st.warning("⚠️ Please enter a YouTube video URL")
            return
        
        video_id = get_video_id(video_url)
        if not video_id:
            st.error("❌ Invalid YouTube URL. Please check the URL and try again.")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("Scraping comments..."):
            df, metadata = scrape_with_progress(
                api_key=youtube_api_key,
                video_url=video_url,
                include_replies=False,
                progress_bar=progress_bar,
                status_text=status_text
            )
        
        progress_bar.empty()
        status_text.empty()

        if df is not None and metadata is not None:
            # Auto-save the file
            saved_filepath = auto_save_to_folder(df, video_id, save_folder)
            
            st.success(f"✅ Scraping completed and saved!")
            st.info(f"📁 **Saved to:** `{saved_filepath}`")
            
            # Store in session state
            st.session_state['latest_csv_path'] = saved_filepath
            st.session_state['latest_df'] = df
            st.session_state['latest_metadata'] = metadata
            
            # Display video metadata
            st.subheader("📊 Video Information")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("👀 Views", f"{int(metadata['view_count']):,}" if metadata['view_count'] != 'N/A' else 'N/A')
            with col2:
                st.metric("👍 Likes", f"{int(metadata['like_count']):,}" if metadata['like_count'] != 'N/A' else 'N/A')
            with col3:
                st.metric("💬 Comments", f"{int(metadata['comment_count']):,}" if metadata['comment_count'] != 'N/A' else 'N/A')
            with col4:
                st.metric("✅ Scraped", f"{len(df):,}")
            
            st.markdown(f"**📹 Title:** {metadata['title']}")
            st.markdown(f"**📺 Channel:** {metadata['channel']}")
            
            # Preview comments
            with st.expander("👀 Preview Comments"):
                st.dataframe(df.head(10))

    # === PART 2: CLUSTER & ANALYZE ===
    st.markdown("---")
    st.header("🧠 Step 2: Cluster & Analyze with AI")
    
    if 'latest_df' not in st.session_state:
        st.info("👆 Please scrape comments first before analyzing.")
        return
    
    # Cluster settings
    col1, col2 = st.columns([1, 3])
    with col1:
        n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=4, 
                               help="Group comments into this many opinion clusters")
    
    if st.button("🔬 Analyze & Cluster Comments", type="primary", use_container_width=True):
        df = st.session_state['latest_df'].copy()
        video_title = st.session_state['latest_metadata']['title']
        
        # Progress tracking
        progress_container = st.container()
        
        with progress_container:
            # Step 1: Filter short comments
            status = st.empty()
            status.info("🔍 Filtering short comments...")
            
            df_filtered = df[df['comment_text'].apply(has_min_words)].copy()
            df_filtered = df_filtered.reset_index(drop=True)
            
            st.success(f"✅ Filtered: {len(df_filtered)} comments remaining (removed {len(df) - len(df_filtered)} short comments)")
            
            # Step 2: Clean comments
            status.info("🧹 Cleaning comments...")
            df_clean = clean_improved_comments(df_filtered)
            st.success("✅ Comments cleaned!")
            
            # Step 3: Generate embeddings
            status.info("🤖 Generating AI embeddings (this may take a minute)...")
            embeddings = generate_embeddings(df_clean['comment_clean'].tolist(), show_progress=False)
            st.success(f"✅ Generated embeddings: {embeddings.shape}")
            
            # Step 4: Perform clustering
            status.info(f"📊 Clustering into {n_clusters} groups...")
            clusters = perform_clustering(embeddings, n_clusters)
            df_clean['cluster'] = clusters
            st.success(f"✅ Clustered into {n_clusters} groups!")
            
            # Store clustered data
            st.session_state['clustered_df'] = df_clean
            st.session_state['n_clusters'] = n_clusters
            
            status.empty()
        
        # Display cluster distribution
        st.subheader("📊 Cluster Distribution")
        cluster_counts = df_clean['cluster'].value_counts().sort_index()
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(cluster_counts.to_frame().rename(columns={'cluster': 'Comment Count'}))
        with col2:
            st.bar_chart(cluster_counts)
        
        # === PART 3: AI SUMMARIES ===
        st.markdown("---")
        st.header("✨ Step 3: AI-Powered Cluster Summaries")
        
        summary_progress = st.progress(0)
        
        for cluster_id in range(n_clusters):
            # Get comments for this cluster
            cluster_df = df_clean[df_clean['cluster'] == cluster_id]
            cluster_comments = cluster_df['comment_text'].tolist()  # Use original text for summarization
            
            # Update progress
            summary_progress.progress((cluster_id + 1) / n_clusters)
            
            # Generate summary
            with st.spinner(f"Summarizing Cluster {cluster_id}..."):
                summary = summarize_comment_cluster(
                    comments=cluster_comments,
                    cluster_id=cluster_id,
                    video_title=video_title
                )
            
            # Display cluster
            with st.expander(f"🎯 Cluster {cluster_id} ({len(cluster_comments)} comments)", expanded=True):
                st.markdown(summary)
                
                st.markdown("**Sample Comments:**")
                for i, comment in enumerate(cluster_comments[:3], 1):
                    st.markdown(f"{i}. *{comment[:200]}{'...' if len(comment) > 200 else ''}*")
        
        summary_progress.empty()
        st.success("🎉 Analysis complete! All clusters summarized.")
        
        # Download option
        st.markdown("---")
        if st.button("💾 Download Clustered Data"):
            csv = df_clean.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"clustered_comments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main()