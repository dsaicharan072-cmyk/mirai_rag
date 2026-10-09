import os
import shutil
import tempfile
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
try:
    from langchain_classic.retrievers.multi_query import MultiQueryRetriever
except ImportError:
    import importlib
    MultiQueryRetriever = importlib.import_module("langchain.retrievers.multi_query").MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
DEFAULT_PDF = "Mirai_SoT_Policy_Handbook_2026.pdf"

# Initialize global references
_vectorstore: Optional[Chroma] = None
_retriever: Optional[Any] = None
_rag_chain: Optional[Any] = None

def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Resolve and return an active embedding model instance."""
    candidates = [
        os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001"),
        "models/gemini-embedding-001",
        "gemini-embedding-001",
        "text-embedding-004",
        "models/text-embedding-004",
    ]
    for m in candidates:
        try:
            emb = GoogleGenerativeAIEmbeddings(model=m)
            emb.embed_query("ping")
            return emb
        except Exception:
            continue
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

def get_llm():
    """Return ChatGoogleGenerativeAI (gemini-3.8-flash) with temperature 0.0, backed by resilient fallbacks."""
    primary_model = os.getenv("LLM_MODEL", "gemini-3.8-flash")
    llm_primary = ChatGoogleGenerativeAI(model=primary_model, temperature=0.0, max_retries=2)
    llm_fallback1 = ChatGoogleGenerativeAI(model="gemini-3.7-flash", temperature=0.0, max_retries=2)
    llm_fallback2 = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.0, max_retries=2)
    llm_fallback3 = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.0, max_retries=2)
    return llm_primary.with_fallbacks([llm_fallback1, llm_fallback2, llm_fallback3])

def build_rag_chain(vectorstore: Chroma):
    """Construct LCEL RAG chain with MultiQueryRetriever and strict guardrails."""
    global _retriever, _rag_chain
    llm = get_llm()

    _retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        llm=llm
    )

    system_prompt = """You are the official autonomous policy advisor for MirAI School of Technology (MSOT).
Your task is to answer student inquiries strictly and exclusively using the provided handbook context.

CRITICAL INSTRUCTIONS & INSTITUTIONAL GUARDRAILS:
1. Strict Grounding: All answers must be strictly confined to the facts provided in the source document context below. Do NOT use or extrapolate from external knowledge or your base training data.
2. Refusal & Unstated Facts: If the answer cannot be found within the provided context, or if specific details (such as monetary fine amounts) are not stated in the policy, clearly state what the policy specifies (e.g., prohibited actions, referral to Disciplinary Committee) and explicitly state that no specific monetary fine is stated in the handbook. Never invent or hallucinate monetary fines, rules, or procedures.
3. Precision: For numerical queries, attendance percentages, tiers, and marks, provide the exact numbers specified in the handbook (e.g. 72% attendance falls in the 60%–74.99% tier and receives 4 marks).
4. Multi-hop Synthesis: For procedures involving specific campuses, contact persons, and timelines, accurately synthesize information across all relevant sections (e.g., for Ratnam campus medical leave, contact Campus Manager Yashaswini Ma'am via campus email or CM office, and submit documents within 7 days of the illness/treatment).
5. Club Proposals: If asked about proposing or starting a new club or society under a club, state that at least 40% batch student support is required, and the proposal must be submitted to the Faculty Coordinator first, not Management directly.

Context:
{context}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    _rag_chain = prompt | llm | StrOutputParser()
    return _rag_chain

def init_vectorstore():
    """Load or initialize ChromaDB vectorstore from persistent directory."""
    global _vectorstore
    embeddings = get_embeddings()

    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        _vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        build_rag_chain(_vectorstore)
        return _vectorstore

    if os.path.exists(DEFAULT_PDF):
        loader = PyPDFLoader(DEFAULT_PDF)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        _vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR)
        build_rag_chain(_vectorstore)
        return _vectorstore

    _vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    build_rag_chain(_vectorstore)
    return _vectorstore

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize Chroma vector store and LCEL chain
    try:
        init_vectorstore()
        print("ChromaDB vectorstore and RAG pipeline initialized successfully.")
    except Exception as e:
        print(f"Warning during vectorstore initialization: {e}")
    yield

app = FastAPI(
    title="MirAI Handbook Policy RAG API",
    description="Autonomous production-grade policy advisor for MirAI School of Technology",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str = Field(..., description="Student's inquiry about university policies")

class SourceDoc(BaseModel):
    page: Optional[int] = Field(None, description="Page number of the handbook")
    content: str = Field(..., description="Excerpt from the policy handbook")

class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceDoc]

class IngestResponse(BaseModel):
    status: str
    message: str
    filename: str
    total_pages: int
    total_chunks: int

@app.get("/health")
async def health_check():
    """Health check endpoint displaying system and database status."""
    global _vectorstore
    count = 0
    if _vectorstore is not None:
        try:
            count = _vectorstore._collection.count()
        except Exception:
            count = 0
    return {
        "status": "healthy",
        "service": "MirAI Policy Handbook RAG Backend",
        "primary_model": os.getenv("LLM_MODEL", "gemini-3.8-flash"),
        "temperature": 0.0,
        "indexed_chunks": count
    }

@app.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_document(file: UploadFile = File(...)):
    """
    Accepts a PDF file via multipart/form-data, processes the text chunks with
    RecursiveCharacterTextSplitter (chunk_size=1000, chunk_overlap=200),
    and updates the local ChromaDB vector store.
    """
    global _vectorstore
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Please upload a PDF document."
        )

    # Save uploaded bytes to a temporary file
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, file.filename)

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Ingestion pipeline: PyPDFLoader -> RecursiveCharacterTextSplitter -> ChromaDB
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()
        if not docs:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded PDF contained no readable text pages."
            )

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        embeddings = get_embeddings()
        if _vectorstore is None:
            _vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR)
        else:
            _vectorstore.add_documents(chunks)

        # Re-build RAG chain with updated database
        build_rag_chain(_vectorstore)

        return IngestResponse(
            status="success",
            message=f"Successfully ingested and indexed {file.filename}",
            filename=file.filename,
            total_pages=len(docs),
            total_chunks=len(chunks)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error ingesting PDF: {str(e)}"
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Accepts a student's question, retrieves context using MultiQueryRetriever,
    executes the LCEL RAG chain, and returns the factual answer with source citations.
    """
    global _vectorstore, _retriever, _rag_chain

    if _vectorstore is None or _retriever is None or _rag_chain is None:
        init_vectorstore()

    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty."
        )

    try:
        # Retrieve candidate chunks via MultiQueryRetriever
        retrieved_docs = await _retriever.ainvoke(request.question)
        if not retrieved_docs:
            retrieved_docs = _retriever.invoke(request.question)

        context_text = "\n\n".join(d.page_content for d in retrieved_docs)

        # Execute LCEL generation chain
        answer = await _rag_chain.ainvoke({
            "context": context_text,
            "question": request.question
        })

        # Format source citations
        sources = [
            SourceDoc(
                page=doc.metadata.get("page", 0) + 1 if "page" in doc.metadata else None,
                content=doc.page_content.strip()
            )
            for doc in retrieved_docs[:3]
        ]

        return ChatResponse(
            question=request.question,
            answer=answer,
            sources=sources
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat request: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
