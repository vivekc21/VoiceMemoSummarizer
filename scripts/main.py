# scripts/main_process.py

import os
import json
from pathlib import Path
import shutil
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI
os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin"
import whisper

load_dotenv()



# Paths
ICLOUD_FOLDER = Path(os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/VoiceMemosToProcess"))
ARCHIVE_FOLDER = ICLOUD_FOLDER / "archive"
ARCHIVE_FOLDER.mkdir(exist_ok=True)

PROCESSED_LOG = Path("'/Users/vivekchinimilli/Library/Mobile Documents/com~apple~CloudDocs/VoiceMemosToProcess/outputs'")

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load Whisper model once
whisper_model = whisper.load_model("base")

def load_processed_files():
    if PROCESSED_LOG.exists():
        with open(PROCESSED_LOG, "r") as f:
            return set(json.load(f))
    return set()

def save_processed_files(processed_files):
    with open(PROCESSED_LOG, "w") as f:
        json.dump(sorted(list(processed_files)), f, indent=2)

def is_valid_audio(file_path: Path):
    return file_path.suffix.lower() in [".mp3", ".m4a", ".wav", ".flac"] and file_path.stat().st_size > 0

def transcribe_audio(file_path: Path) -> str:
    print(f"🔊 Transcribing locally: {file_path.name}")
    result = whisper_model.transcribe(str(file_path), fp16=False)
    return result["text"]

def summarize_transcript(transcript: str) -> str:
    print(f"🧠 Summarizing transcript for {len(transcript)} chars")
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes meeting transcripts."},
            {"role": "user", "content": f"Here is a transcript of a meeting:\n\n{transcript}\n\nWrite concise meeting minutes in markdown format summarizing the key outcomes and action items from my discussion"}
        ],
        temperature=0.3,
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()

def save_summary(file_path: Path, summary: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = Path('/Users/vivekchinimilli/Library/Mobile Documents/com~apple~CloudDocs/VoiceMemosToProcess/outputs')
    output_folder.mkdir(exist_ok=True)
    summary_file = output_folder / f"{file_path.stem}_summary_{timestamp}.md"
    with open(summary_file, "w") as f:
        f.write(summary)
    print(f"✅ Saved summary: {summary_file}")

def move_to_archive(file_path: Path):
    dest = ARCHIVE_FOLDER / file_path.name
    shutil.move(str(file_path), str(dest))
    print(f"📂 Moved {file_path.name} to archive.")

def main():
    processed_files = load_processed_files()
    new_files = [f for f in ICLOUD_FOLDER.iterdir() if f.is_file() and is_valid_audio(f) and f.name not in processed_files]

    if not new_files:
        print("No new audio files found.")
        return

    for file_path in new_files:
        try:
            transcript = transcribe_audio(file_path)
            summary = summarize_transcript(transcript)
            save_summary(file_path, summary)
            move_to_archive(file_path)

            processed_files.add(file_path.name)
            save_processed_files(processed_files)
        except Exception as e:
            print(f"❌ Error processing {file_path.name}: {e}")

if __name__ == "__main__":
    main()