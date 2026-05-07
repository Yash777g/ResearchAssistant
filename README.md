# 🔬 Document-Aware GenAI Research Assistant

> Final Year Project | Python · LangChain · FAISS · Ollama · FastAPI

A fully local, offline-capable AI research assistant that lets you upload research papers
and interact with them via Q&A, summarization, **research gap detection**, and **auto literature review generation**.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 PDF Upload & Processing | Upload multiple PDFs — chunked, embedded, stored in FAISS |
| 💬 RAG-based Q&A | Ask questions; get answers with page-level citations |
| 📝 Paper Summarization | Academic / Bullet / Brief styles using map-reduce |
| 🔍 **Research Gap Detection** | AI identifies unexplored areas, limitations, future directions |
| 📚 **Auto Literature Review** | Generates structured academic review from your papers |

---

## 🛠 Tech Stack

- **Backend**: FastAPI + Uvicorn
- **LLM**: Ollama (llama3.2 — runs fully offline)
- **Embeddings**: nomic-embed-text via Ollama
- **Vector Store**: FAISS (local, no cloud needed)
- **RAG Framework**: LangChain
- **PDF Parsing**: PyPDF
- **Frontend**: Vanilla HTML/CSS/JS (served by FastAPI)

---

## 🚀 Setup & Run

### Step 1 — Install Ollama
```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: Download from https://ollama.com/download
```

### Step 2 — Pull Required Models
```bash
ollama pull llama3.2           # LLM (~2GB)
ollama pull nomic-embed-text   # Embeddings (~270MB)
```

### Step 3 — Clone & Install Python Dependencies
```bash
git clone <your-repo-url>
cd genai-research-assistant

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Step 4 — Start Ollama Server
```bash
ollama serve
# Runs at http://localhost:11434
```

### Step 5 — Run the Application
```bash
uvicorn app.main:app --reload --port 8000
```

### Step 6 — Open in Browser
```
http://localhost:8000
```

---

## 📁 Project Structure

```
genai-research-assistant/
├── app/
│   ├── main.py                  # FastAPI entry point
│   ├── api/
│   │   ├── documents.py         # Upload / list / reset PDFs
│   │   ├── qa.py                # RAG-based Q&A
│   │   ├── summarize.py         # Paper summarization
│   │   └── research.py          # Gap detection + Literature review ⭐
│   ├── core/
│   │   └── config.py            # LLM, embeddings, config
│   ├── services/
│   │   ├── pdf_service.py       # PDF loading & chunking
│   │   └── vector_store.py      # FAISS operations
│   └── static/
│       └── index.html           # Frontend UI
├── uploads/                     # Uploaded PDFs
├── vectorstore/                 # FAISS index files
├── requirements.txt
├── .env                         # Model config
└── README.md
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/documents/upload` | Upload PDFs |
| GET | `/api/documents/list` | List uploaded docs |
| DELETE | `/api/documents/reset` | Clear all docs |
| POST | `/api/qa/ask` | Ask a question (RAG) |
| POST | `/api/summarize/summarize` | Summarize a paper |
| POST | `/api/research/gap-detection` | **Detect research gaps** |
| POST | `/api/research/literature-review` | **Generate literature review** |

Full interactive API docs: `http://localhost:8000/docs`

---

## 💡 Innovation Details

### 🔍 Research Gap Detection
Analyzes uploaded papers to identify:
- Topics mentioned as "future work" or "limitations"
- Contradictions / unresolved debates between papers
- Assumptions never validated
- Practical applications not yet explored

### 📚 Auto Literature Review
Generates a full structured academic review with:
- Introduction & background
- Key themes synthesis (across all papers)
- Methodology comparison
- Synthesized findings
- Gap analysis
- Conclusion & future directions

---

## ⚙️ Change LLM Model

Edit `.env`:
```env
OLLAMA_MODEL=mistral        # or phi3, gemma2, deepseek-r1
EMBED_MODEL=nomic-embed-text
```
Then restart the server.

---

## 📊 For College Evaluation

- ✅ Fully local — no API keys or internet needed during demo
- ✅ Real RAG pipeline (not simulated)
- ✅ FAISS vector store with persistent storage
- ✅ 2 unique innovations (Gap Detection + Literature Review)
- ✅ Clean REST API with auto-generated Swagger docs at `/docs`
- ✅ Modular, well-structured Python codebase
