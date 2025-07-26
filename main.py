
import os
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL")

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

db = Chroma(
    persist_directory='chroma_db',
    embedding_function=embeddings
)

qa = RetrievalQA.from_chain_type(
    llm=OllamaLLM(model="mistral"),
    retriever=db.as_retriever(),
    return_source_documents=True
)

print("🟢 QA System Ready. Type 'exit' to quit.\n")

while True:
    query = input("❓ Your question: ")
    if query.strip().lower() in {"exit", "quit"}:
        print("👋 Exiting. Goodbye!")
        break

    try:
        result = qa.invoke(query)
        print("\n📘 Answer:\n", result["result"])
        print("\n📄 Source Docs:\n", [doc.metadata for doc in result["source_documents"]])
    except Exception as e:
        print("❌ Error:", e)