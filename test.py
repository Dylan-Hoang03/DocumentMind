import requests

res = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "mistral",
        "prompt": "What is FAISS?",
        "stream": False
    },
    timeout=60
)
print(res.json())
