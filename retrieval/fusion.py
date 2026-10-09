import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger("LegalRAG.Retrieval.Fusion")


class ReciprocalRankFusion:
    """Reciprocal Rank Fusion (RRF) for merging BM25 and Dense candidate pools."""

    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k

    def fuse_rankings(
        self,
        bm25_results: List[Tuple[Dict[str, Any], float]],
        dense_results: List[Tuple[Dict[str, Any], float]],
        top_k: int = 50
    ) -> List[Tuple[Dict[str, Any], float]]:
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}

        # 1. Process BM25 rankings
        for rank, (chunk, score) in enumerate(bm25_results, 1):
            cid = chunk["chunk_id"]
            chunk_map[cid] = chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        # 2. Process Dense rankings
        for rank, (chunk, score) in enumerate(dense_results, 1):
            cid = chunk["chunk_id"]
            chunk_map[cid] = chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        # 3. Sort fused candidates by RRF score descending
        fused_items = [(chunk_map[cid], rrf_scores[cid]) for cid in rrf_scores]
        fused_items.sort(key=lambda x: x[1], reverse=True)

        logger.info(f"RRF Fusion (k={self.rrf_k}): Merged {len(bm25_results)} BM25 and {len(dense_results)} Dense candidates into {len(fused_items[:top_k])} hybrid candidates.")
        return fused_items[:top_k]

