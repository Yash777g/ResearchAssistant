import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List

from app.core.config import UPLOAD_DIR
from app.services.pdf_service import process_pdf
from app.services.vector_store import load_doc_metadata, reset_vectorstore

router = APIRouter()


@router.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload one or more PDF research papers."""
    results = []

    for file in files:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{file.filename} is not a PDF.")

        save_path = os.path.join(UPLOAD_DIR, file.filename)

        # Save file to disk
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process and embed
        try:
            pages, chunks = process_pdf(save_path, file.filename)
            results.append({
                "filename": file.filename,
                "pages": pages,
                "chunks": chunks,
                "status": "success"
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "failed",
                "error": str(e)
            })

    return JSONResponse(content={"uploaded": results})


@router.get("/list")
async def list_documents():
    """List all uploaded documents."""
    docs = load_doc_metadata()
    return {"documents": docs, "count": len(docs)}


@router.delete("/reset")
async def reset_all():
    """Delete all uploaded documents and reset the vector store."""
    # Clear uploads folder
    for f in os.listdir(UPLOAD_DIR):
        fp = os.path.join(UPLOAD_DIR, f)
        if os.path.isfile(fp):
            os.remove(fp)

    # Reset FAISS index
    reset_vectorstore()

    return {"message": "All documents and vector store cleared successfully."}
