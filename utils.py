import os
import cohere
from dotenv import load_dotenv
import requests


load_dotenv()
cohere_api_key = os.getenv("COHERE_API_KEY")
client = cohere.Client(cohere_api_key)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# OpenRouter API endpoint for chat completions
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Function to send a request to OpenRouter
def call_openrouter(prompt):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "anthropic/claude-3-haiku",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0.4
    }

    response = requests.post(OPENROUTER_URL, json=payload, headers=headers)

    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()  # Extract the response
    else:
        return f"Error: {response.status_code}, {response.text}"