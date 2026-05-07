from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

from app.core.config import get_llm, get_embeddings
from app.services.vector_store import get_vectorstore, retrieve_relevant_chunks

router = APIRouter()


class QuestionRequest(BaseModel):
    question: str
    top_k: int = 6


QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an expert research assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say "I could not find this in the uploaded documents."
Always cite which part of the document supports your answer.

Context:
{context}

Question: {question}

Answer (with citations):"""
)


@router.post("/ask")
async def ask_question(request: QuestionRequest):
    """Ask a question about the uploaded documents using RAG."""
    vectorstore = get_vectorstore()
    if not vectorstore:
        raise HTTPException(
            status_code=404,
            detail="No documents uploaded yet. Please upload PDFs first."
        )

    # Retrieve relevant chunks
    chunks = retrieve_relevant_chunks(request.question, k=request.top_k)
    if not chunks:
        return {"answer": "No relevant content found.", "sources": []}

    # Build context from chunks
    context = "\n\n---\n\n".join([
        f"[Source: {c.metadata.get('source', 'Unknown')}, Page {c.metadata.get('page', '?')}]\n{c.page_content}"
        for c in chunks
    ])

    # Run LLM
    llm    = get_llm()
    prompt = QA_PROMPT.format(context=context, question=request.question)

    try:
        answer = llm.invoke(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

    # Build source references
    sources = list({
        f"{c.metadata.get('source', 'Unknown')} — Page {c.metadata.get('page', '?')}"
        for c in chunks
    })

    return {
        "question": request.question,
        "answer": answer,
        "sources": sources,
        "chunks_used": len(chunks)
    }
