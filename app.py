from flask import Flask, request, Response, jsonify
from qa_core import init_qa_system, stream_answer

app = Flask(__name__)
init_qa_system()

@app.route("/")
def index():
    return "LLM QA System is running."

@app.route("/api/ask_stream", methods=["POST"])
def ask_stream():
    data = request.get_json()
    query = data.get("question", "").strip()
    if not query:
        return jsonify({"error": "Missing question field"}), 400
    return Response(stream_answer(query), content_type="text/plain")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)