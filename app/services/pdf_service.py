import os
from typing import List, Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from app.core.config import get_text_splitter, UPLOAD_DIR
from app.services.vector_store import add_documents_to_store, save_doc_metadata


def process_pdf(file_path: str, filename: str) -> Tuple[int, int]:
    """
    Load a PDF, split into chunks, embed and store in FAISS.
    Returns (num_pages, num_chunks).
    """
    # Load PDF
    loader = PyPDFLoader(file_path)
    pages: List[Document] = loader.load()

    # Add source metadata to each page
    for page in pages:
        page.metadata["source"] = filename

    # Split into chunks
    splitter = get_text_splitter()
    chunks   = splitter.split_documents(pages)

    # Store in FAISS
    add_documents_to_store(chunks)
    save_doc_metadata([filename])

    return len(pages), len(chunks)


def get_full_text_from_pdf(file_path: str) -> str:
    """Extract full raw text from a PDF (used for summarization)."""
    loader = PyPDFLoader(file_path)
    pages  = loader.load()
    return "\n\n".join([p.page_content for p in pages])
