# scripts/setup_test.py

import openai
import requests
import schedule
from dotenv import load_dotenv
import os

def test_setup():
    print("All libraries imported successfully.")
    load_dotenv()
    if os.getenv("OPENAI_API_KEY"):
        print("API key found in .env (we'll test it in Step 2).")
    else:
        print("No API key found yet in .env.")

if __name__ == "__main__":
    test_setup()