#!/usr/bin/env python3
# scripts/main_process.py

import os
import json
from pathlib import Path
import shutil
from datetime import datetime
import numpy as np

from dotenv import load_dotenv
from openai import OpenAI
os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin"
import whisper
from audio_utils import load_audio_robust, preprocess_audio_for_whisper

load_dotenv()



# Paths
ICLOUD_FOLDER = Path(os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/VoiceMemosToProcess"))
ARCHIVE_FOLDER = ICLOUD_FOLDER / "archive"
ARCHIVE_FOLDER.mkdir(exist_ok=True)

PROCESSED_LOG = Path('/Users/vivekchinimilli/Library/Mobile Documents/com~apple~CloudDocs/VoiceMemosToProcess/outputs/processed_files.json')

# OpenRouter client (compatible with OpenAI API)
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
)

# Load Whisper model once
whisper_model_name = os.getenv("WHISPER_MODEL", "base")
print(f"🔧 Loading Whisper model: {whisper_model_name}")
whisper_model = whisper.load_model(whisper_model_name)

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
    print(f"📍 File path: {file_path}")
    print(f"📏 File size: {file_path.stat().st_size} bytes")
    print(f"🎵 File extension: {file_path.suffix}")
    
    try:
        # Check if file exists and is readable
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
            
        if file_path.stat().st_size == 0:
            raise ValueError(f"Audio file is empty: {file_path}")
        
        # Load audio using robust method
        print(f"🔧 Loading audio with robust method...")
        audio = load_audio_robust(file_path, target_sr=16000)
        
        if audio is None:
            raise RuntimeError("Failed to load audio with all available methods")
        
        # Preprocess audio for Whisper (limit to ~30 minutes to prevent memory issues)
        max_samples = 30 * 60 * 16000  # 30 minutes at 16kHz
        audio = preprocess_audio_for_whisper(audio, target_length=max_samples)
        
        print(f"🔧 Audio loaded successfully: {len(audio)} samples ({len(audio)/16000:.1f} seconds)")
        print(f"🔧 Transcribing with Whisper model...")
        
        # Transcribe using preprocessed audio
        result = whisper_model.transcribe(
            audio,
            fp16=False,
            verbose=False,  # Reduce verbosity to avoid clutter
            language='en',  # Specify language if known
            word_timestamps=False  # Disable word timestamps to save memory
        )
        
        print(f"✅ Transcription completed successfully")
        return result["text"]
        
    except Exception as e:
        print(f"❌ Detailed transcription error: {type(e).__name__}: {str(e)}")
        raise

def clean_model_output(content: str) -> str:
    """Clean model output by removing thinking tags and extracting clean summary."""
    if not content:
        return ""
    
    # Remove thinking tags (common in some models)
    import re
    
    # Remove <think>...</think> blocks
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    
    # Remove <reasoning>...</reasoning> blocks  
    content = re.sub(r'<reasoning>.*?</reasoning>', '', content, flags=re.DOTALL)
    
    # Remove other common thinking patterns
    content = re.sub(r'<thought>.*?</thought>', '', content, flags=re.DOTALL)
    
    # Clean up extra whitespace and newlines
    content = re.sub(r'\n\s*\n', '\n\n', content)  # Multiple newlines to double
    content = content.strip()
    
    return content

def summarize_with_fallback_models(transcript: str) -> str:
    """Try multiple models in order of preference until one succeeds."""
    
    # Models to try in order (most preferred first)
    models_to_try = [
        os.getenv("OPENROUTER_MODEL", "moonshotai/kimi-k2:free"),  # User's preferred model first
        "openai/gpt-4o-mini",
        "openai/gpt-3.5-turbo",
        "anthropic/claude-3-haiku", 
        "meta-llama/llama-3-8b-instruct:free",
    ]
    
    for i, model in enumerate(models_to_try):
        print(f"🧪 Trying model {i+1}/{len(models_to_try)}: {model}")
        
        try:
            result = summarize_with_model(transcript, model)
            if result and not result.startswith("ERROR:") and len(result) > 50:
                print(f"✅ Success with model: {model}")
                return result
            else:
                print(f"⚠️  Model {model} returned insufficient content")
                
        except Exception as e:
            print(f"❌ Model {model} failed: {str(e)}")
            continue
    
    return "ERROR: All summarization models failed"

def chunk_transcript(transcript: str, max_chunk_size: int = 8000) -> list:
    """Split transcript into chunks for processing long content."""
    if len(transcript) <= max_chunk_size:
        return [transcript]
    
    chunks = []
    words = transcript.split()
    current_chunk = []
    current_size = 0
    
    for word in words:
        word_size = len(word) + 1  # +1 for space
        if current_size + word_size > max_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_size = word_size
        else:
            current_chunk.append(word)
            current_size += word_size
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    print(f"📄 Split transcript into {len(chunks)} chunks")
    return chunks

def summarize_with_model(transcript: str, model: str) -> str:
    """Summarize transcript with a specific model."""
    
    # Handle very long transcripts by chunking
    if len(transcript) > 10000:
        print(f"📄 Long transcript detected ({len(transcript)} chars), chunking...")
        chunks = chunk_transcript(transcript, max_chunk_size=8000)
        
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            print(f"🔄 Processing chunk {i+1}/{len(chunks)}...")
            chunk_summary = summarize_single_chunk(chunk, model, i+1)
            if chunk_summary and not chunk_summary.startswith("ERROR:"):
                chunk_summaries.append(chunk_summary)
        
        if chunk_summaries:
            # Combine chunk summaries into final summary
            combined_text = "\n\n".join(chunk_summaries)
            print(f"🔄 Creating final summary from {len(chunk_summaries)} chunks...")
            return summarize_single_chunk(
                f"Here are summaries of different parts of a long meeting:\n\n{combined_text}\n\nPlease create a comprehensive meeting summary combining these parts.",
                model,
                "final"
            )
        else:
            return "ERROR: Failed to process any chunks"
    else:
        return summarize_single_chunk(transcript, model, 1)

def summarize_single_chunk(transcript: str, model: str, chunk_num) -> str:
    """Summarize a single chunk or full transcript."""
    
    print(f"🧠 Summarizing chunk {chunk_num} ({len(transcript)} chars) with {model}")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant that summarizes meeting transcripts. Provide clean, concise summaries in markdown format without any thinking or reasoning tags."},
        {"role": "user", "content": f"Here is a transcript of a meeting:\n\n{transcript}\n\nWrite concise meeting minutes in markdown format summarizing the key outcomes and action items from this discussion. Do not include any thinking, reasoning, or meta-commentary - just the clean summary."}
    ]
    
    print(f"🌐 Making API request...")
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=1000,
    )
    
    print(f"✅ Received API response")
    
    if hasattr(response, 'choices') and len(response.choices) > 0:
        choice = response.choices[0]
        if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
            content = choice.message.content
            print(f"📄 Raw content length: {len(content) if content else 0}")
            
            if content and content.strip():
                # Clean the content to remove thinking tags
                cleaned_content = clean_model_output(content)
                
                print(f"🧹 Cleaned content length: {len(cleaned_content)} chars (was {len(content)})")
                
                if cleaned_content and len(cleaned_content) > 50:
                    return cleaned_content
                else:
                    return content.strip()  # Return original if cleaning removed too much
            else:
                return "ERROR: Empty content in API response"
        else:
            return "ERROR: Invalid response format from OpenRouter API"
    else:
        return "ERROR: No choices in OpenRouter API response"

def summarize_transcript(transcript: str) -> str:
    """Main summarization function with fallback models and error handling."""
    print(f"🧠 Summarizing transcript for {len(transcript)} chars")
    print(f"📝 Transcript preview: {transcript[:200]}...")
    
    try:
        # Use fallback models approach for better reliability
        result = summarize_with_fallback_models(transcript)
        
        if result and not result.startswith("ERROR:"):
            print(f"✅ Summarization completed successfully ({len(result)} chars)")
            return result
        else:
            print(f"❌ All summarization attempts failed: {result}")
            return result
            
    except Exception as e:
        print(f"❌ Critical error during summarization: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"ERROR: Critical summarization failure - {type(e).__name__}: {str(e)}"

def save_summary(file_path: Path, summary: str):
    print(f"💾 Saving summary for {file_path.name}")
    print(f"📝 Summary content length: {len(summary)} chars")
    print(f"📝 Summary preview: {summary[:100]}...")
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = Path('/Users/vivekchinimilli/Library/Mobile Documents/com~apple~CloudDocs/VoiceMemosToProcess/outputs')
        
        print(f"📁 Output folder: {output_folder}")
        print(f"📁 Output folder exists: {output_folder.exists()}")
        
        # Create output folder
        output_folder.mkdir(exist_ok=True, parents=True)
        print(f"📁 Output folder created/verified")
        
        summary_file = output_folder / f"{file_path.stem}_summary_{timestamp}.md"
        print(f"📄 Target file: {summary_file}")
        
        # Write summary with error checking
        with open(summary_file, "w", encoding='utf-8') as f:
            f.write(summary)
            f.flush()  # Ensure content is written
        
        # Verify the file was written correctly
        if summary_file.exists():
            file_size = summary_file.stat().st_size
            print(f"✅ File written successfully: {summary_file}")
            print(f"📊 File size: {file_size} bytes")
            
            # Read back content to verify
            with open(summary_file, "r", encoding='utf-8') as f:
                saved_content = f.read()
                print(f"🔍 Verification: Read back {len(saved_content)} chars")
                
                if len(saved_content) == len(summary):
                    print(f"✅ File content verified successfully")
                else:
                    print(f"⚠️  Warning: Content length mismatch - saved {len(saved_content)}, expected {len(summary)}")
        else:
            print(f"❌ Error: File was not created at {summary_file}")
        
        return summary_file
        
    except Exception as e:
        print(f"❌ Error saving summary: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Try backup location
        try:
            backup_file = Path(f"/tmp/{file_path.stem}_summary_{timestamp}.md")
            print(f"💾 Trying backup location: {backup_file}")
            with open(backup_file, "w", encoding='utf-8') as f:
                f.write(summary)
            print(f"✅ Backup saved: {backup_file}")
            return backup_file
        except Exception as backup_error:
            print(f"❌ Backup save also failed: {backup_error}")
            return None

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