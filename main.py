import os

import assemblyai as aai
import dotenv

dotenv.load_dotenv()

assemblyai.settings.api_key = os.getenv("ASSEMBLYAI_TOKEN")


config = aai.TranscriptionConfig(iab_categories=True)

transcript = aai.Transcriber().transcribe(audio_url, config)

# Get the parts of the transcript that were tagged with topics
for result in transcript.iab_categories.results:
    print(result.text)
    print(f"Timestamp: {result.timestamp.start} - {result.timestamp.end}")
    for label in result.labels:
        print(f"{label.label} ({label.relevance})")

# Get a summary of all topics in the transcript
for topic, relevance in transcript.iab_categories.summary.items():
    print(f"Audio is {relevance * 100}% relevant to {topic}")
