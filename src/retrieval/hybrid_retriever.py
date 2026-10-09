from typing import Dict, List, Tuple, Optional
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever


class HybridRetriever:
    """
    Hybrid Retriever combining lexical (BM25) and dense semantic (MiniLM-L6-v2) candidate pools
    using Reciprocal Rank Fusion (RRF).
    
    RRF Score Formula:
        RRF_score(d) = sum_{r in retrievers} (1.0 / (k + rank_r(d)))
    """
    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        rrf_k: int = 60,
        bm25_top_k: int = 50,
        dense_top_k: int = 50
    ):
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.rrf_k = rrf_k
        self.bm25_top_k = bm25_top_k
        self.dense_top_k = dense_top_k

    def rank_documents(self, query_text: str, top_k: int = 50) -> List[Tuple[str, float]]:
        """
        Executes BM25 & Dense retrieval, fuses candidate ranks via RRF, and returns top_k results.
        """
        if not query_text or not query_text.strip():
            return []

        # 1. Retrieve candidates from lexical BM25 & dense retrievers
        bm25_results = self.bm25_retriever.rank_documents(query_text, top_k=self.bm25_top_k)
        dense_results = self.dense_retriever.rank_documents(query_text, top_k=self.dense_top_k)

        # 2. Accumulate Reciprocal Rank Fusion (RRF) scores
        rrf_scores: Dict[str, float] = {}

        for rank, (doc_id, _score) in enumerate(bm25_results, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (self.rrf_k + rank))

        for rank, (doc_id, _score) in enumerate(dense_results, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (self.rrf_k + rank))

        # 3. Sort candidates by RRF score descending
        fused_candidates = list(rrf_scores.items())
        fused_candidates.sort(key=lambda x: x[1], reverse=True)

        top_k = min(top_k, len(fused_candidates))
        return fused_candidates[:top_k]

