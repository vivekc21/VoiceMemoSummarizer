# scripts/audio_processor.py

import os
import json
from pathlib import Path
from datetime import datetime

ICLOUD_AUDIO_FOLDER = os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/VoiceMemosToProcess")
PROCESSED_LOG = Path("/Users/vivekchinimilli/voice-summary-project/processed/processed_files.json")

def load_processed_files():
    if PROCESSED_LOG.exists():
        with open(PROCESSED_LOG, "r") as f:
            return set(json.load(f))
    return set()

def save_processed_files(processed_files):
    with open(PROCESSED_LOG, "w") as f:
        json.dump(sorted(list(processed_files)), f, indent=2)

def is_valid_audio(file_path):
    if not file_path.suffix.lower() == ".m4a":
        return False
    if file_path.stat().st_size == 0:
        return False
    return True

def scan_for_new_audio():
    processed_files = load_processed_files()
    new_files = []

    for file_path in Path(ICLOUD_AUDIO_FOLDER).glob("*.m4a"):
        if file_path.name not in processed_files and is_valid_audio(file_path):
            print(f"✅ New valid audio file found: {file_path.name}")
            new_files.append(file_path)

    return new_files, processed_files

def update_processed_log(new_files, processed_files):
    for file_path in new_files:
        processed_files.add(file_path.name)
    save_processed_files(processed_files)

if __name__ == "__main__":
    new_files, processed_files = scan_for_new_audio()
    update_processed_log(new_files, processed_files)
    print(f"{len(new_files)} new files logged.")
