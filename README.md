# YT to Blog

CLI tool that takes a YouTube URL, fetches the transcript, and asks a local Ollama model to turn it into a podcast-style script saved as Markdown.

## What It Does

- Extracts a YouTube video ID from common YouTube URL formats
- Downloads the transcript with `youtube-transcript-api`
- Builds a prompt for Ollama
- Generates rewritten output from the configured local model
- Saves the result to a filename derived from the generated title

## Setup

```bash
uv sync
```

Create a `.env` file with the Ollama model name:

```env
OLLAMA_MODEL_NAME=llama3.2
```

Make sure Ollama is running and the model exists locally.

## Run

```bash
uv run python main.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

## Notes

- Supported URL formats include `watch`, `youtu.be`, `embed`, and `shorts`.
- The current prompt generates a podcast conversation, not a traditional blog post.
- Tests live in `test_main.py`.
