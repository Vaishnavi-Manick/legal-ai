import os
import sys
import time
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ingest.pipeline import load_config
from retrieval.query_engine import Phase3QueryEngine
from src.inference import LegalSearchEngine
from src.llm.rag_pipeline import LegalRAGPipeline

logger = logging.getLogger("LegalAIBackend")
logging.basicConfig(level=logging.INFO)


class Phase3QueryEngineAdapter:
    """Adapts Phase3QueryEngine to provide search(query_text, top_k) interface expected by LegalRAGPipeline."""

    def __init__(self, config: Dict[str, Any] = None):
        self.engine = Phase3QueryEngine(config)

    def search(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        res = self.engine.search(query_text)
        formatted = []
        for rank, (chunk, score) in enumerate(res.get("final_top_5", [])[:top_k], 1):
            formatted.append({
                "rank": rank,
                "case_id": chunk["case_id"],
                "relevance_score": round(float(score), 4),
                "case_name": chunk.get("case_name") or f"Case {chunk['case_id']}",
                "citation": f"AILA 2019 / {chunk.get('section_type', 'doc')}",
                "snippet": chunk["text"][:300] + "..." if len(chunk["text"]) > 300 else chunk["text"],
                "document_text": chunk["text"]
            })
        return formatted


app = FastAPI(
    title="Legal AI RAG Case Retrieval API",
    description="FastAPI backend for Phase 3/5 Hybrid Legal Case Retrieval & Grounded RAG (AILA 2019)",
    version="2.0.0"
)

# CORS configuration for React development server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global LegalRAGPipeline instance - loaded ONCE at application startup
rag_pipeline: Optional[LegalRAGPipeline] = None


@app.on_event("startup")
def startup_event():
    global rag_pipeline
    logger.info("Initializing Phase 3 Hybrid Query Engine and LegalRAGPipeline...")
    try:
        config = load_config("config.yaml")
        hybrid_adapter = Phase3QueryEngineAdapter(config)
        rag_pipeline = LegalRAGPipeline(search_engine=hybrid_adapter)
        logger.info("LegalRAGPipeline with Phase 3 Hybrid Query Engine initialized successfully.")
    except Exception as e:
        logger.warning(f"Could not initialize Phase 3 Hybrid Engine ({e}). Falling back to V1 LegalSearchEngine...", exc_info=True)
        v1_engine = LegalSearchEngine(candidate_pool_size=50)
        rag_pipeline = LegalRAGPipeline(search_engine=v1_engine)
        logger.info("LegalRAGPipeline initialized with V1 LegalSearchEngine fallback.")


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5


class CaseResult(BaseModel):
    rank: int
    case_id: str
    relevance_score: float
    case_name: Optional[str] = None
    citation: Optional[str] = None
    snippet: str
    document_text: str


class SearchResponse(BaseModel):
    query: str
    search_time: float
    results: List[CaseResult]
    answer: Optional[str] = None
    notice: Optional[str] = None
    supporting_cases: Optional[List[Dict[str, Any]]] = None
    disclaimer: Optional[str] = None
    llm_available: bool = False
    latency: Optional[Dict[str, float]] = None


@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Legal AI Case Retrieval & Grounded RAG API",
        "dataset": "AILA 2019",
        "docs_count": 2914,
        "engine_active": "Phase 3 Hybrid Query Engine (BM25 + Dense + RRF + V1 MaxP Reranker)",
        "llm_available": rag_pipeline.llm_client.is_available() if rag_pipeline else False
    }


@app.post("/search", response_model=SearchResponse)
def search_cases(request: SearchRequest):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")

    if rag_pipeline is None:
        raise HTTPException(status_code=500, detail="Legal RAG pipeline is not initialized.")

    query = request.query.strip()
    top_k = request.top_k or 5

    try:
        pipeline_output = rag_pipeline.run_pipeline(query_text=query, top_k=top_k)

        results = []
        for r in pipeline_output.get("results", []):
            results.append(CaseResult(
                rank=r["rank"],
                case_id=r["case_id"],
                relevance_score=r["relevance_score"],
                case_name=r.get("case_name"),
                citation=r.get("citation"),
                snippet=r["snippet"],
                document_text=r["document_text"]
            ))

        latency = pipeline_output.get("latency", {})
        search_time = latency.get("total_time", 0.0)

        return SearchResponse(
            query=query,
            search_time=search_time,
            results=results,
            answer=pipeline_output.get("answer"),
            notice=pipeline_output.get("notice"),
            supporting_cases=pipeline_output.get("supporting_cases"),
            disclaimer=pipeline_output.get("disclaimer"),
            llm_available=pipeline_output.get("llm_available", False),
            latency=latency
        )
    except Exception as e:
        logger.error(f"Error during RAG search execution: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during legal search: {str(e)}")
