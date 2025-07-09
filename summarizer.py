# scripts/summarizer.py

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def summarize_transcript(transcript: str) -> str:
    try:
        print("🧠 Sending transcript to ChatGPT for summarization...")

        response = client.chat.completions.create(
            model="gpt-4",  # Or "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes meeting transcripts."},
                {"role": "user", "content": f"Here is a transcript of a meeting:\n\n{transcript}\n\nPlease summarize the key points and action items."}
            ],
            temperature=0.3,
            max_tokens=500,
        )

        summary = response.choices[0].message.content.strip()
        print("✅ Summarization complete.")
        return summary

    except Exception as e:
        print(f"❌ Error during summarization: {e}")
        return None

if __name__ == "__main__":
    test_transcript = """
    Hi everyone, let's get started. First item: product launch. Sarah said we’re on track but marketing needs assets by next Friday. 
    Budget: $10K approved for Q3 campaign. Offsite: Tahoe is frontrunner.
    """
    summary = summarize_transcript(test_transcript)
    print("\nSUMMARY:\n", summary)
