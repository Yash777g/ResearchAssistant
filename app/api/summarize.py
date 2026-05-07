import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document
from langchain.prompts import PromptTemplate

from app.core.config import get_llm, get_text_splitter, UPLOAD_DIR
from app.services.pdf_service import get_full_text_from_pdf

router = APIRouter()


class SummarizeRequest(BaseModel):
    filename: str
    style: str = "academic"   # academic | bullet | brief


SUMMARY_PROMPTS = {
    "academic": """Write a comprehensive academic summary of the following research paper.
Include: Objective, Methodology, Key Findings, Contributions, and Limitations.

Text:
{text}

Academic Summary:""",

    "bullet": """Summarize the following research paper as bullet points.
Cover: main objective, methods used, key results, and conclusions.

Text:
{text}

Bullet Point Summary:""",

    "brief": """Write a 3-5 sentence plain-language summary of this research paper
suitable for a non-expert audience.

Text:
{text}

Brief Summary:"""
}


@router.post("/summarize")
async def summarize_document(request: SummarizeRequest):
    """Summarize an uploaded PDF research paper."""
    file_path = os.path.join(UPLOAD_DIR, request.filename)

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail=f"File '{request.filename}' not found. Please upload it first."
        )

    style = request.style if request.style in SUMMARY_PROMPTS else "academic"

    # Extract full text
    try:
        full_text = get_full_text_from_pdf(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF read error: {str(e)}")

    # Split into chunks for map-reduce summarization
    splitter = get_text_splitter()
    chunks   = splitter.create_documents([full_text])

    # Use map-reduce chain for long documents
    llm = get_llm()

    map_prompt = PromptTemplate(
        input_variables=["text"],
        template="Summarize this section of the research paper concisely:\n\n{text}\n\nSection Summary:"
    )
    combine_prompt = PromptTemplate(
        input_variables=["text"],
        template=SUMMARY_PROMPTS[style]
    )

    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=combine_prompt,
        verbose=False
    )

    try:
        result = chain.invoke({"input_documents": chunks})
        summary = result.get("output_text", "Summary could not be generated.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization error: {str(e)}")

    return {
        "filename": request.filename,
        "style": style,
        "summary": summary,
        "total_chunks": len(chunks)
    }
