from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from app.api import documents, qa, research, summarize

app = FastAPI(
    title="Document-Aware GenAI Research Assistant",
    description="Upload research papers and get AI-powered Q&A, gap detection, literature review, and summaries.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Include routers
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
app.include_router(qa.router, prefix="/api/qa", tags=["Q&A"])
app.include_router(research.router, prefix="/api/research", tags=["Research"])
app.include_router(summarize.router, prefix="/api/summarize", tags=["Summarize"])

@app.get("/")
async def serve_frontend():
    return FileResponse("app/static/index.html")

@app.get("/health")
async def health_check():
    return {"status": "running", "message": "GenAI Research Assistant is live"}
