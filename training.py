import os
import hashlib
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
TRAIN_MODE = os.getenv("TRAIN_MODE", "full")
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
DATA_PATH = os.getenv("DATA_PATH", "data")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL")
HASH_FILE = "trained_hashes.txt"

def file_hash(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def load_trained_hashes():
    if not os.path.exists(HASH_FILE):
        return set()
    with open(HASH_FILE, "r") as f:
        return set(line.strip() for line in f.readlines())
    
def save_trained_hashes(hashes):
    with open(HASH_FILE, "w") as f:
        for h in hashes:
            f.write(h + "\n")

def main():
    trained_hashes = load_trained_hashes()
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

    if TRAIN_MODE == "full":
        if os.path.exists(CHROMA_PATH):
            import shutil
            shutil.rmtree(CHROMA_PATH)
        if os.path.exists(HASH_FILE):
            os.remove(HASH_FILE)
        trained_hashes = set()
    
    loader = DirectoryLoader(
        path="data",
        glob='**/*',
        loader_cls=lambda path: {
            ".txt": TextLoader,
            ".pdf": PyPDFLoader,
        }.get(os.path.splitext(path)[1], TextLoader)(path)
    )

    all_documents = loader.load()

    new_documents = []
    new_hashes = set()

    for doc in all_documents:
        path = doc.metadata.get("source")
        if not path:
            continue
        h = file_hash(path)
        if h not in trained_hashes:
            new_documents.append(doc)
            new_hashes.add(h)

    if not new_documents:
        print("No new documents to train.")
        return
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )
    doc = splitter.split_documents(new_documents)

    db = None
    if os.path.exists(CHROMA_PATH) and os.path.isdir(CHROMA_PATH):
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        print(f"Loaded existing Chroma database at {CHROMA_PATH}.")
        db.add_documents(doc)
    else:
        print(f"No existing Chroma database found at {CHROMA_PATH}. Creating a new one.")
        db = Chroma.from_documents(doc, embeddings, persist_directory=CHROMA_PATH)

    trained_hashes.update(new_hashes)
    save_trained_hashes(trained_hashes)
    
    print("✅ Training completed and stored in 'chroma_db'.")

main()