# scripts/transcriber.py

import whisper
import openai
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
model = whisper.load_model("turbo")

def transcribe_audio(file_path: Path) -> str:
    print(f"🔊 Transcribing {file_path.name}...")
    try:
        result = model.transcribe(str(file_path))
        print(f"✅ Local transcription complete.")
        return result["text"]
    except Exception as e:
        print(f"❌ Local Whisper error: {e}")
        return None

if __name__ == "__main__":
    # Quick test block
    test_file = Path("~/Library/Mobile Documents/com~apple~CloudDocs/VoiceMemosToProcess/sample.m4a").expanduser()
    if test_file.exists():
        transcript = transcribe_audio(test_file)
        if transcript:
            print("TRANSCRIPT:\n", transcript[:500])
    else:
        print("Drop a sample file in the iCloud folder and try again.")
