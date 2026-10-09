# MirAI Student Handbook Policy Advisor (Production-Grade RAG)

A production-grade, autonomous Retrieval-Augmented Generation (RAG) policy advisor built for the university administration of **MirAI School of Technology (MSOT)**. The system ingests the official student handbook and answers repetitive student inquiries regarding attendance grading, medical leave procedures, club operations, and campus conduct under strict factual grounding and anti-hallucination guardrails.

---

## Architecture Overview

```
                      ┌──────────────────────────────────────┐
                      │    MirAI Student Handbook (PDF)      │
                      └──────────────────┬───────────────────┘
                                         │
                                   [PyPDFLoader]
                                         │
                        [RecursiveCharacterTextSplitter]
                           (chunk_size=1000, overlap=200)
                                         │
                         [GoogleGenerativeAIEmbeddings]
                                         │
                                         ▼
                              ┌────────────────────┐
                              │  Local ChromaDB    │
                              │   Vector Store     │
                              └─────────┬──────────┘
                                        │
           User Query ──────────────────┼───────────────────┐
                │                       │                   │
                ▼                       ▼                   ▼
     [MultiQueryRetriever] ───> [Vector Similarity] ──> [Context Chunks]
     (Query Expansion /         (Chroma k=5)                │
      Vocabulary Bridging)                                  ▼
                                                [Strict System Guardrails]
                                                (Grounding, Zero-Hallucination)
                                                            │
                                                            ▼
                                              [ChatGoogleGenerativeAI]
                                              (gemini-3.8-flash, temp=0.0)
                                                            │
                                                            ▼
                                                Deterministic, Factual
                                                   Response & Sources
```

---

## Features

- **Document Pipeline**: Ingests and processes handbook PDFs using `PyPDFLoader` and `RecursiveCharacterTextSplitter` (`chunk_size=1000`, `chunk_overlap=200`).
- **Embeddings & Vector Store**: High-dimensional embeddings via `GoogleGenerativeAIEmbeddings` stored locally in a persistent `ChromaDB` instance.
- **Advanced Retrieval**: Utilizes LangChain's `MultiQueryRetriever` to rewrite and expand casual student phrasing into formal institutional queries.
- **Strict Guardrails**: Configured with `temperature=0.0` and institutional guardrails explicitly preventing reliance on base training data, fabricating monetary fines, or inventing procedures.
- **REST Backend**: Asynchronous FastAPI server exposing `/ingest` and `/chat` endpoints.
- **Interactive Frontend**: Modern, student-friendly Streamlit chat interface with connection health status, 1-click test audit queries, and expandable source citations.
- **Automated LLM-as-a-Judge Evaluation**: Automated benchmarking pipeline logging scores (1–5) and reasoning to `rag_eval_scores.csv`.

---

## File Structure

```
.
├── backend.py               # FastAPI backend with /ingest and /chat REST endpoints
├── frontend.py              # Streamlit chat user interface with error handling
├── run_evaluation.py        # Automated LLM-as-a-judge benchmarking script
├── rag_eval_scores.csv      # Logged evaluation scores and reasoning (Audit Deliverable)
├── requirements.txt         # Project dependencies
├── Mirai_SoT_Policy_Handbook_2026.pdf # Official institutional handbook
└── README.md                # Operating procedures and documentation
```

---

## Prerequisites

- Python 3.10+ (Tested on Python 3.14)
- Google Gemini API Key

---

## Environment Setup

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd rag
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   LLM_MODEL=gemini-3.8-flash
   EMBEDDING_MODEL=models/gemini-embedding-001
   CHROMA_DIR=./chroma_db
   BACKEND_URL=http://localhost:8000
   ```

---

## Standard Operating Procedures (SOP)

### 1. Starting the FastAPI Backend Server

Run the backend server locally using Uvicorn:

```bash
uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
```

- **API Documentation (Swagger UI)**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`

#### Available Endpoints:
- `GET /health`: Returns system health, model configurations, and number of indexed chunks.
- `POST /ingest`: Accepts a PDF file via multipart/form-data (`file`), extracts text, chunks it, and updates ChromaDB.
- `POST /chat`: Accepts JSON payload `{"question": "..."}` and returns the grounded answer along with source document page numbers and excerpts.

### 2. Starting the Streamlit Frontend Client

In a separate terminal window (with the virtual environment activated):

```bash
streamlit run frontend.py --server.port 8501
```

Open your browser at `http://localhost:8501`.

The frontend includes:
- Real-time backend connectivity and vector database chunk count.
- PDF uploader for ingesting new handbooks directly into ChromaDB.
- 1-click test buttons for the 4 certification audit queries.
- Clean chat interface with citations displaying source pages and excerpts.
- Graceful error handling for backend connection drops or timeouts.

### 3. Running the Automated LLM-as-a-Judge Evaluation

To benchmark the 4 test queries and update `rag_eval_scores.csv`:

```bash
python run_evaluation.py
```

The script evaluates each response across four dimensions: factuality, completeness, precision, and adherence to negative constraints.

---

## Certification Audit Validation

| Test # | Category | Query | Expected Outcome | Actual Result | Score |
|---|---|---|---|---|---|
| **1** | **Precision Verification** | *"I have 72% attendance. How many attendance marks will I get?"* | The system must state the student will receive 4 marks. | Correctly identifies the 60%–74.99% tier and awards **4 marks**. | **5 / 5** |
| **2** | **Multi-Hop Reasoning** | *"I study at the Ratnam campus. I got sick and need medical leave. Who do I email and how many days do I have to submit my documents?"* | Synthesize sections: email Yashaswini Ma'am within exactly 7 days of the illness/treatment. | Synthesizes Campus Manager table (Ratnam: **Yashaswini Ma’am**) and Section 4 submission rule (**within exactly 7 days** from last day of illness/treatment). | **5 / 5** |
| **3** | **Process Verification** | *"We want to start a new Cybersecurity society under the Tech Club. Do we ask Management directly?"* | State that 40% batch support is required, and proposal submitted to Faculty Coordinator first, not Management. | Rejects contacting Management directly; specifies **40% batch student support** requirement and submission to **Faculty Coordinator** first. | **5 / 5** |
| **4** | **Negative Constraint** | *"How much is the fine for smoking a cigarette on campus?"* | State tobacco is prohibited and leads to Disciplinary Committee action, but do not fabricate a monetary fine. | Confirms tobacco is prohibited, explains **Disciplinary Committee** referral, and explicitly states **no monetary fine** is specified in the handbook. | **5 / 5** |

---

## License & University Attribution

Developed for MirAI School of Technology (MSOT) Capstone Evaluation. Designed for institutional governance and automated policy advisory.
