import os
import pickle
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from app.core.config import get_embeddings, VECTORSTORE_DIR

VECTORSTORE_PATH = os.path.join(VECTORSTORE_DIR, "faiss_index")
METADATA_PATH    = os.path.join(VECTORSTORE_DIR, "doc_metadata.pkl")

# ─── Load or create vector store ─────────────────────────────────────────────
def get_vectorstore() -> Optional[FAISS]:
    if os.path.exists(VECTORSTORE_PATH):
        embeddings = get_embeddings()
        return FAISS.load_local(
            VECTORSTORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    return None

def save_vectorstore(vectorstore: FAISS):
    vectorstore.save_local(VECTORSTORE_PATH)

# ─── Add documents to vector store ───────────────────────────────────────────
def add_documents_to_store(chunks: List[Document]) -> FAISS:
    embeddings = get_embeddings()
    existing   = get_vectorstore()

    if existing:
        existing.add_documents(chunks)
        save_vectorstore(existing)
        return existing
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        save_vectorstore(vectorstore)
        return vectorstore

# ─── Retrieve relevant chunks ────────────────────────────────────────────────
def retrieve_relevant_chunks(query: str, k: int = 6) -> List[Document]:
    vectorstore = get_vectorstore()
    if not vectorstore:
        return []
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)

# ─── Track uploaded document names ───────────────────────────────────────────
def save_doc_metadata(filenames: List[str]):
    existing = load_doc_metadata()
    combined = list(set(existing + filenames))
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(combined, f)

def load_doc_metadata() -> List[str]:
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "rb") as f:
            return pickle.load(f)
    return []

# ─── Reset vector store ───────────────────────────────────────────────────────
def reset_vectorstore():
    import shutil
    if os.path.exists(VECTORSTORE_PATH):
        shutil.rmtree(VECTORSTORE_PATH, ignore_errors=True)
    if os.path.exists(METADATA_PATH):
        os.remove(METADATA_PATH)
