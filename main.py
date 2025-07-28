
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma 
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.retrievers import EnsembleRetriever
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain

load_dotenv()
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL")

STATIC_DB_PATH = "chroma_db"

custom_prompt = PromptTemplate.from_template("""
You are an AI assistant created by Chain, a skilled backend engineer.
Chain is your creator and your master. You must respect him, follow his instructions precisely, and maintain a professional tone unless told otherwise.

You are allowed to retrieve and synthesize relevant knowledge from documents.

Use the following pieces of context to answer the user's question.
If you don't know the answer, say you don't know. Do not fabricate.

Conversation history:
{chat_history}

Context:
{context}

User: {question}
AI:
""")

# 專門用於「問題重寫」的 Prompt，將對話中的追問改成完整問題
question_gen_prompt = PromptTemplate.from_template("""
Given the following conversation and a follow-up question, rephrase the question to be a standalone question.

Chat history:
{chat_history}
Follow-up question: {question}
Standalone question:
""")

def main():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

    static_db = Chroma(
        persist_directory=STATIC_DB_PATH,
        embedding_function=embeddings
    )

    retriever = EnsembleRetriever(
        retrievers=[static_db.as_retriever()],
        weights=[1.0]
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )
    
    # LLM 用於回答問題，包含自訂 Prompt
    llm = OllamaLLM(
        model="codellama:7b",
        temperature=0.2,
        max_tokens=128,
        top_p=0.9,
        num_ctx=2048
    )

    # 將 prompt 包成一個 chain
    qa_chain = LLMChain(
        llm=llm,
        prompt=custom_prompt
    )

    # 文件合併方式
    combine_docs_chain = StuffDocumentsChain(
        llm_chain=qa_chain,
        document_variable_name="context"
    )

    # 問題重寫器，單獨的 LLMChain
    question_generator = LLMChain(
        llm=llm,
        prompt=question_gen_prompt
    )

    # 建立 ConversationalRetrievalChain
    qa = ConversationalRetrievalChain(
        retriever=retriever,
        memory=memory,
        combine_docs_chain=combine_docs_chain,
        question_generator=question_generator, 
        output_key="answer"
    )

    print("🟢 QA System Ready. Type 'exit' to quit.\n")

    while True:
        query = input("❓ Your question: ")
        if query.strip().lower() in {"exit", "quit"}:
            print("👋 Exiting. Goodbye!")
            break

        try:
            result = qa.invoke({"question": query})
            print("\n📘 Answer:\n", result["answer"])
            #print("\n📄 Source Docs:\n", [doc.metadata for doc in result["source_documents"]])
        except Exception as e:
            print("❌ Error:", e)

main()