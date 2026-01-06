import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=GEMINI_API_KEY)

# Load model once
model = genai.GenerativeModel("gemini-2.5-flash")


def summarize_comment_cluster(comments, cluster_id, video_title=None):
    """
    Summarize one opinion cluster of YouTube comments.

    Args:
        comments (list[str]): All comments belonging to one cluster
        cluster_id (int): Cluster number
        video_title (str, optional): Video title for context

    Returns:
        str: Gemini-generated summary
    """

    if not comments:
        return f"Cluster {cluster_id}: No comments available."

    # Select representative comments (top 15 longest)
    selected_comments = sorted(comments, key=len, reverse=True)[:15]
    comments_text = "\n- ".join(selected_comments)

    prompt = f"""
You are an objective researcher analyzing public opinion on YouTube.

CONTEXT:
Video Title: {video_title if video_title else "N/A"}
Cluster ID: {cluster_id}

TASK:
The following comments belong to ONE opinion cluster.
Summarize the shared theme and viewpoint expressed in this group.

RULES:
- neutral tone
- No usernames
- No hallucination
- Base conclusions strictly on the comments

OUTPUT FORMAT:
Theme: <short descriptive label>
Summary: <2-3 sentences>
Keywords: <3-5 keywords>

COMMENTS:
- {comments_text}
"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating summary for cluster {cluster_id}: {e}"