import requests

response = requests.post(
    "http://localhost:5000/api/ask_stream",
    json={"question": "What did it include?"},
    stream=True
)
print("Answer:")
print()
for chunk in response.iter_content(chunk_size=None):
    print(chunk.decode(), end="")