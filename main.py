import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.prompts import PromptTemplate

load_dotenv()
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL")

STATIC_DB_PATH = "chroma_db"  # Static knowledge vector DB path
MEMORY_DB_PATH = "memory_db"  # Dialogue memory vector DB path

PROMPT = os.getenv("PROMPT")

custom_prompt = PromptTemplate.from_template(PROMPT)

def store_dialogue_to_vectorstore(user_input, ai_response, vectorstore):
    # Save user and AI conversation as a Document with timestamp metadata
    content = f"User: {user_input}\nAI: {ai_response}"
    doc = Document(
        page_content=content,
        metadata={
            "source": "dialogue_memory",  # Mark as dialogue memory source
            "timestamp": datetime.now().isoformat(),  # Timestamp for sorting
        }
    )
    vectorstore.add_documents([doc])  # Add to vectorstore (embedding + persist)

def load_recent_chat_history(vectorstore, limit=6):
    # Retrieve recent conversation turns from vectorstore by similarity
    docs = vectorstore.similarity_search("dialogue_memory", k=limit)
    sorted_docs = sorted(docs, key=lambda d: d.metadata.get("timestamp", ""))
    return [doc.page_content for doc in sorted_docs]

def main():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    static_db = Chroma(
        persist_directory=STATIC_DB_PATH,
        embedding_function=embeddings
    )
    memory_db = Chroma(
        persist_directory=MEMORY_DB_PATH,
        embedding_function=embeddings
    )

    bm25_texts = static_db.get()["documents"]
    bm25_metadatas = static_db.get()["metadatas"]
    bm25_retriever = BM25Retriever.from_texts(
        texts=bm25_texts,
        metadatas=bm25_metadatas
    )
    bm25_retriever.k = 3  # Number of top docs to retrieve by BM25

    llm = OllamaLLM(
        model="codellama:7b",
        temperature=0.2,
        max_tokens=128,
        top_p=0.9,
        num_ctx=2048,
        streaming=True  # Enable streaming token output
    )

    print("🟢 QA System (LLM streaming) Ready. Type 'exit' to quit.\n")

    chat_history = load_recent_chat_history(memory_db, limit=6)  # Load persistent chat history

    while True:
        query = input("❓ Your question: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("👋 Exiting. Goodbye!")
            break

        if query.lower().startswith("recall:"):
            keyword = query.split(":", 1)[1].strip()
            recall_results = memory_db.similarity_search(keyword, k=3)  # Recall from memory by keyword
            print("\n🧠 Memory Recall:")
            for i, doc in enumerate(recall_results):
                print(f"#{i+1}:\n{doc.page_content}")
            continue

        dense_docs = static_db.similarity_search(query, k=1)  # Retrieve from static knowledge DB
        memory_docs = memory_db.similarity_search(query, k=1)  # Retrieve from dialogue memory DB
        bm25_docs = bm25_retriever.invoke(query)  # Sparse retrieval using BM25

        context = ""
        for doc in dense_docs + memory_docs + bm25_docs:
            context += doc.page_content + "\n---\n"  # Combine retrieved docs into context string

        chat_history_str = "\n".join(chat_history[-6:])  # Use last 6 turns of chat history

        prompt = custom_prompt.format(
            chat_history=chat_history_str,
            context=context,
            question=query
        )

        print("\n📘 Answer:\n", end="", flush=True)
        response = ""
        for token in llm.stream(prompt):  # Stream LLM output token-by-token
            print(token, end="", flush=True)
            response += token
        print()

        store_dialogue_to_vectorstore(query, response, memory_db)  # Save Q&A to persistent memory
        chat_history.append(f"User: {query}\nAI: {response}")  # Append to in-memory chat history

main()
