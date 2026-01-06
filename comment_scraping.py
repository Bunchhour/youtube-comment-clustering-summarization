import os
import re
import time
from dotenv import load_dotenv
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from datetime import datetime

# Load environment variables
load_dotenv()

def get_video_id(url):
    """Extracts the video ID from a YouTube URL."""
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11})"
    match = re.search(regex, url)
    return match.group(1) if match else None

def get_video_metadata(youtube, video_id):
    """Fetch video metadata."""
    try:
        request = youtube.videos().list(
            part="snippet,statistics",
            id=video_id
        )
        response = request.execute()

        if not response.get('items'):
            print("⚠ No video metadata found.")
            return None

        video = response['items'][0]
        return {
            'title': video['snippet']['title'],
            'channel': video['snippet']['channelTitle'],
            'published_at': video['snippet']['publishedAt'],
            'view_count': video['statistics'].get('viewCount', 'N/A'),
            'like_count': video['statistics'].get('likeCount', 'N/A'),
            'comment_count': video['statistics'].get('commentCount', 'N/A')
        }
    except Exception as e:
        print(f"❌ Error fetching metadata: {e}")
        return None

def scrape_youtube_comments(api_key, video_url, delay_between_requests=0.5):
    """Scrape top-level comments from a YouTube video."""
    video_id = get_video_id(video_url)
    if not video_id:
        print("❌ Invalid YouTube URL.")
        return None

    youtube = build('youtube', 'v3', developerKey=api_key)

    print("Fetching video metadata...")
    metadata = get_video_metadata(youtube, video_id)
    if metadata:
        print(f"Video: {metadata['title']}")
        print(f"Channel: {metadata['channel']}")
        print(f"Reported comments: {metadata['comment_count']}\n")

    comments_data = []
    next_page_token = None
    page_count = 0

    print(f"Starting scrape for Video ID: {video_id}...\n")

    try:
        while True:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token,
                textFormat="plainText"
            )
            response = request.execute()
            page_count += 1

            for item in response.get('items', []):
                snippet = item['snippet']['topLevelComment']['snippet']
                comments_data.append({
                    'comment_id': item['id'],
                    'comment_text': snippet['textOriginal'],
                    'author': snippet['authorDisplayName'],
                    'like_count': snippet.get('likeCount', 0),
                    'published_at': snippet['publishedAt']
                })

            next_page_token = response.get('nextPageToken')

            print(f"Page {page_count}: Collected {len(comments_data)} comments")

            if not next_page_token:
                break

            time.sleep(delay_between_requests)

        df = pd.DataFrame(comments_data)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"youtube_comments_{video_id}_{timestamp}.csv"
        df.to_csv(filename, index=False, encoding="utf-8-sig")

        print("\n" + "=" * 50)
        print("SCRAPING COMPLETE")
        print("=" * 50)
        print(f"Total comments collected: {len(df)}")
        print(f"Saved to: {filename}")
        print("=" * 50 + "\n")

        return df

    except HttpError as e:
        print(f"❌ YouTube API Error: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return None

def main():
    print("\n" + "=" * 60)
    print(" " * 18 + "YouTube Comment Scraper")
    print("=" * 60 + "\n")

    api_key = os.getenv("YouTube_API")

    if not api_key:
        print("❌ ERROR: YouTube API key not found.")
        print("Add to .env: YouTube_API=YOUR_API_KEY")
        return

    print(f"✓ API Key loaded: {api_key[:10]}...{api_key[-4:]}\n")

    while True:
        video_url = input("Enter YouTube URL (or 'q' to quit): ").strip()
        if video_url.lower() in ['q', 'quit', 'exit']:
            print("👋 Goodbye!")
            break

        if not get_video_id(video_url):
            print("❌ Invalid YouTube URL.\n")
            continue

        scrape_youtube_comments(api_key, video_url)

        again = input("\nScrape another video? (Y/n): ").strip().lower()
        if again == 'n':
            print("👋 Goodbye!")
            break
        print()

if __name__ == "__main__":
    main()