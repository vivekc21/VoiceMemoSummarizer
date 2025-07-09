# scripts/test_openai_api.py

import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def test_openai():
    try:
        models = openai.models.list()
        print("OpenAI API connection successful. Models available:")
        for model in models.data[:5]:  # print a few for verification
            print(f"- {model.id}")
    except Exception as e:
        print("Error connecting to OpenAI API:")
        print(e)

if __name__ == "__main__":
    test_openai()