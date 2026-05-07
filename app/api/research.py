from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain.prompts import PromptTemplate

from app.core.config import get_llm
from app.services.vector_store import get_vectorstore, retrieve_relevant_chunks

router = APIRouter()


# ══════════════════════════════════════════════════════
# 1. RESEARCH GAP DETECTION
# ══════════════════════════════════════════════════════

class GapDetectionRequest(BaseModel):
    topic: str = ""       # Optional: focus area. If empty, uses all docs.
    num_gaps: int = 5


GAP_DETECTION_PROMPT = PromptTemplate(
    input_variables=["context", "topic", "num_gaps"],
    template="""You are a senior research analyst specializing in identifying gaps in academic literature.

Analyze the following excerpts from research papers carefully.
{topic_instruction}

Your task is to identify {num_gaps} clear RESEARCH GAPS — areas that:
- Have NOT been studied or addressed in these papers
- Are mentioned as "future work" or "limitations"
- Show contradictions or unresolved debates between papers
- Are assumptions made but never validated
- Are practical applications not yet explored

Research Paper Excerpts:
{context}

For each research gap, provide:
1. Gap Title (concise name)
2. Description (2-3 sentences explaining the gap)
3. Evidence (which paper/section hints at this gap)
4. Suggested Research Direction (how this gap could be addressed)

Format each gap clearly under a numbered heading.

Research Gaps Identified:"""
)


@router.post("/gap-detection")
async def detect_research_gaps(request: GapDetectionRequest):
    """
    ENHANCEMENT 1: Research Gap Detection
    Analyzes uploaded papers to find unexplored areas, limitations, and future research directions.
    """
    vectorstore = get_vectorstore()
    if not vectorstore:
        raise HTTPException(
            status_code=404,
            detail="No documents uploaded. Please upload research papers first."
        )

    # Build search query
    search_query = request.topic if request.topic else "limitations future work research gaps methodology assumptions"

    # Retrieve relevant chunks (use more chunks for better gap detection)
    chunks = retrieve_relevant_chunks(search_query, k=10)
    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant content found.")

    # Build context
    context = "\n\n---\n\n".join([
        f"[Paper: {c.metadata.get('source', 'Unknown')}, Page {c.metadata.get('page', '?')}]\n{c.page_content}"
        for c in chunks
    ])

    topic_instruction = (
        f"Focus specifically on the topic: '{request.topic}'."
        if request.topic
        else "Analyze across all topics covered in the papers."
    )

    llm    = get_llm()
    prompt = GAP_DETECTION_PROMPT.format(
        context=context,
        topic_instruction=topic_instruction,
        num_gaps=request.num_gaps
    )

    try:
        result = llm.invoke(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

    sources = list({c.metadata.get("source", "Unknown") for c in chunks})

    return {
        "topic": request.topic or "All uploaded papers",
        "gaps_requested": request.num_gaps,
        "research_gaps": result,
        "papers_analyzed": sources,
        "chunks_analyzed": len(chunks)
    }


# ══════════════════════════════════════════════════════
# 2. AUTO LITERATURE REVIEW GENERATION
# ══════════════════════════════════════════════════════

class LiteratureReviewRequest(BaseModel):
    topic: str
    include_sections: list[str] = [
        "introduction",
        "themes",
        "methodologies",
        "findings",
        "gaps",
        "conclusion"
    ]


LITERATURE_REVIEW_PROMPT = PromptTemplate(
    input_variables=["context", "topic", "sections"],
    template="""You are an expert academic writer. Generate a structured literature review
on the topic: "{topic}"

Use ONLY the research paper excerpts provided below as your source material.
Write in a formal academic tone. Synthesize ideas across papers — do not summarize
each paper individually.

Research Paper Excerpts:
{context}

Generate a literature review with the following sections: {sections}

For each section:
- Write 2-4 coherent paragraphs
- Cite papers by their filename when referencing specific claims
- Identify agreements AND disagreements between authors
- Use academic language (e.g., "Several studies suggest...", "Contrary to X, Y argues...")

Literature Review on: {topic}
{"=" * 60}
"""
)

SECTION_PROMPTS = {
    "introduction": "1. INTRODUCTION\nProvide background, importance of the topic, and scope of this review.",
    "themes": "2. KEY THEMES & CONCEPTS\nIdentify and discuss major recurring themes across the papers.",
    "methodologies": "3. METHODOLOGIES\nCompare and contrast the research methods used across the papers.",
    "findings": "4. KEY FINDINGS\nSynthesize the major results and contributions from the papers.",
    "gaps": "5. RESEARCH GAPS & LIMITATIONS\nHighlight what remains unexplored or contested in the literature.",
    "conclusion": "6. CONCLUSION\nSummarize the state of knowledge and suggest future research directions."
}


@router.post("/literature-review")
async def generate_literature_review(request: LiteratureReviewRequest):
    """
    ENHANCEMENT 2: Auto Literature Review Generation
    Generates a structured academic literature review from uploaded papers.
    """
    if not request.topic.strip():
        raise HTTPException(status_code=400, detail="Topic is required.")

    vectorstore = get_vectorstore()
    if not vectorstore:
        raise HTTPException(
            status_code=404,
            detail="No documents uploaded. Please upload research papers first."
        )

    # Retrieve broadly relevant content
    chunks = retrieve_relevant_chunks(request.topic, k=12)
    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant content found for the given topic.")

    context = "\n\n---\n\n".join([
        f"[Paper: {c.metadata.get('source', 'Unknown')}, Page {c.metadata.get('page', '?')}]\n{c.page_content}"
        for c in chunks
    ])

    # Build sections string
    valid_sections = [s for s in request.include_sections if s in SECTION_PROMPTS]
    sections_text  = "\n".join([SECTION_PROMPTS[s] for s in valid_sections])

    llm    = get_llm()
    prompt = LITERATURE_REVIEW_PROMPT.format(
        context=context,
        topic=request.topic,
        sections=sections_text
    )

    try:
        review = llm.invoke(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

    papers = list({c.metadata.get("source", "Unknown") for c in chunks})

    return {
        "topic": request.topic,
        "sections_included": valid_sections,
        "literature_review": review,
        "papers_used": papers,
        "total_chunks": len(chunks)
    }
