# YouTube Transcript to Blog Post Converter
# Main script file

import re
import os
import sys
import argparse
import ollama
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

# Load environment variables from .env file
load_dotenv()

def extract_video_id(video_url: str) -> str | None:
    """
    Extract YouTube video ID from various YouTube URL formats.
    
    Args:
        video_url (str): YouTube URL in various formats
        
    Returns:
        str | None: Video ID if found, None if not a valid YouTube URL or no ID found
        
    Supported formats:
        - youtube.com/watch?v=VIDEO_ID
        - youtu.be/VIDEO_ID
        - youtube.com/embed/VIDEO_ID
        - youtube.com/shorts/VIDEO_ID
    """
    # Define regex patterns for different YouTube URL formats
    patterns = [
        r'(?:youtube\.com/watch\?v=)([a-zA-Z0-9_-]{11})',  # youtube.com/watch?v=VIDEO_ID
        r'(?:youtu\.be/)([a-zA-Z0-9_-]{11})',              # youtu.be/VIDEO_ID
        r'(?:youtube\.com/embed/)([a-zA-Z0-9_-]{11})',     # youtube.com/embed/VIDEO_ID
        r'(?:youtube\.com/shorts/)([a-zA-Z0-9_-]{11})'     # youtube.com/shorts/VIDEO_ID
    ]
    
    # Try each pattern to find a match
    for pattern in patterns:
        match = re.search(pattern, video_url)
        if match:
            return match.group(1)
    
    # Return None if no pattern matches
    return None


def get_youtube_transcript(video_id: str) -> str | None:
    """
    Fetch YouTube video transcript and return as a single string.
    
    Args:
        video_id (str): YouTube video ID
        
    Returns:
        str | None: Transcript text as a single string if successful, 
                   None if transcript not found or error occurs
    """
    try:
        # Fetch transcript from YouTube
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Concatenate all text segments into a single string
        transcript_text = " ".join([segment['text'] for segment in transcript_list])
        
        return transcript_text
        
    except TranscriptsDisabled:
        print(f"Error: Transcripts are disabled for video ID: {video_id}")
        return None
    except NoTranscriptFound:
        print(f"Error: No transcript found for video ID: {video_id}")
        return None
    except Exception as e:
        print(f"Error fetching transcript for video ID {video_id}: {str(e)}")
        return None


def create_podcast_conversation_prompt(transcript_text: str) -> str:
    """
    Formats the transcript into a prompt for an LLM to generate a podcast-style conversation summary.
    """
    prompt_template = '''<System>: You are a skilled summarizer and conversation rewriter, specializing in turning excerpts into engaging podcast conversations. Your role is to analyze any given excerpt, identify clear actionable insights and takeaways, and then reframe them into a casual and engaging podcast-style dialogue. Emphasize accuracy and disclaim if you're unsure about any points. Offer actionable tips, suggestions, and topics of debate based on the excerpt.

<Context>: The user provides an excerpt of text. Your task is to distill key insights and transform them into a dynamic podcast conversation between two hosts. The style should be friendly, informal, and relatable, as if the hosts are chatting naturally. The user expects actionable insights that are accurate and free from hallucination.

<Instructions>:

1️⃣ Read the provided excerpt thoroughly.
2️⃣ Identify and extract the main insights and key takeaways.
3️⃣ Frame these insights into clear, actionable tips and suggestions.
4️⃣ Create a script-like dialogue between two podcast hosts.

* Use natural, flowing conversation that keeps the audience engaged.
* Add casual humor or witty remarks if appropriate, but keep it grounded in the excerpt's content.
* Encourage curiosity and debate by including questions or discussion points.
  5️⃣ If the excerpt contains uncertain data or incomplete information, disclaim it or suggest clarifying questions for the hosts to ask.
  6️⃣ Make sure the conversation is focused and not overly long—aim for brevity while covering all important points.
  7️⃣ Final output: A script-like dialogue with clear tips, suggestions, and interesting debate topics.

<Constraints>:

* Keep language friendly and engaging.
* No made-up data—disclaim if unsure.
* Provide clear, actionable insights.
* Use a script format (like "Host 1: ... Host 2: ...").

<Output Format>:
A script-like podcast dialogue covering:
🔹 Key insights distilled from the excerpt
🔹 Actionable tips and suggestions
🔹 Debatable or curious questions for deeper exploration

---

Here is the excerpt to convert:

{transcript_text}

---

'''
    return prompt_template.format(transcript_text=transcript_text)


def generate_podcast_conversation_with_ollama(prompt: str, model_name: str) -> str | None:
    """
    Sends the prompt to a local Ollama model and returns the generated podcast conversation text.
    """
    try:
        # Make the API call to Ollama
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a skilled summarizer and conversation rewriter, specializing in turning excerpts into engaging podcast conversations."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract the content from the response
        podcast_content = response['message']['content']
        
        return podcast_content
        
    except ollama.ResponseError as e:
        print(f"Error: Ollama API error with model '{model_name}': {str(e)}")
        return None
    except Exception as e:
        print(f"Error: Connection or other error when calling Ollama: {str(e)}")
        return None


def extract_title_from_podcast(podcast_script: str) -> str:
    """
    Extract a title from the podcast script (use first non-empty line or fallback).
    """
    # Try to find a line like 'Host 1: ...' and use the first few words
    for line in podcast_script.splitlines():
        line = line.strip()
        if line and not line.lower().startswith('host 1:') and not line.lower().startswith('host 2:'):
            # Use this as a title
            title = line
            break
    else:
        # Fallback: use first 5 words from the script
        words = podcast_script.strip().split()
        title = "_".join(words[:5]) if words else "podcast"
    # Sanitize
    sanitized_title = re.sub(r'[<>:"/\\|?*]', '', title)
    sanitized_title = re.sub(r'\s+', '_', sanitized_title)
    sanitized_title = sanitized_title[:100]
    if not sanitized_title or sanitized_title.isspace():
        sanitized_title = "podcast"
    return sanitized_title


def main_logic():
    """Main script logic for converting YouTube video to podcast conversation."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Convert YouTube video transcript to podcast conversation using Ollama")
    parser.add_argument("youtube_url", help="YouTube video URL to convert")
    args = parser.parse_args()
    
    # Retrieve OLLAMA_MODEL_NAME from environment variables
    ollama_model = os.getenv("OLLAMA_MODEL_NAME")
    if not ollama_model:
        print("Error: OLLAMA_MODEL_NAME not set in .env or environment. Please set it (e.g., 'llama3.2') and ensure the model is pulled with 'ollama pull <model_name>'.")
        sys.exit(1)
    
    # Use the provided YouTube URL
    youtube_url = args.youtube_url
    print(f"Processing YouTube URL: {youtube_url}")
    
    # Extract video ID
    video_id = extract_video_id(youtube_url)
    if video_id is None:
        print(f"Error: Could not extract video ID from URL: {youtube_url}")
        sys.exit(1)
    
    print(f"Extracted Video ID: {video_id}")
    
    # Fetch transcript
    print(f"Fetching transcript for video ID: {video_id}...")
    transcript_text = get_youtube_transcript(video_id)
    if transcript_text is None:
        print("Transcript not found or error fetching.")
        sys.exit(1)
    
    print("Transcript fetched successfully.")
    
    # Create prompt
    print(f"Creating podcast conversation prompt for Ollama model: {ollama_model}...")
    prompt = create_podcast_conversation_prompt(transcript_text)
    
    # Generate podcast conversation with Ollama
    print("Sending prompt to Ollama...")
    podcast_script = generate_podcast_conversation_with_ollama(prompt, ollama_model)
    if podcast_script is not None:
        print("Podcast conversation generated successfully.")
        
        # Extract title from podcast script and create filename
        title = extract_title_from_podcast(podcast_script)
        output_filename = f"{title}_podcast.md"
        
        # Write podcast conversation to file
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(podcast_script)
        
        print(f"Podcast conversation saved to {output_filename}")
    else:
        print("Failed to generate podcast conversation.")
        sys.exit(1)


if __name__ == "__main__":
    main_logic() 