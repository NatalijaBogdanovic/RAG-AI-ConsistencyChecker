import os
from dotenv import load_dotenv
import cohere

# Učitavanje .env varijabli
load_dotenv()

cohere_api_key = os.getenv("COHERE_API_KEY")
if not cohere_api_key:
    raise ValueError("COHERE_API_KEY nije postavljen! Proveri .env fajl.")

client = cohere.Client(cohere_api_key)

test_query = ["Ovo je test."]
response = client.embed(
    texts=test_query,
    model="embed-english-v3.0",
    truncate="NONE",
    input_type="search_query"  # Dodato ovo
)
print(response.embeddings)
