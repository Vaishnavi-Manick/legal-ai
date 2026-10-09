import os
import sys
import time
import logging
from typing import Dict, Any, List, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ingest.pipeline import load_config
from index.dense_indexer import DenseIndexer
from index.bm25_indexer import BM25Indexer
from retrieval.router import CitationRouter
from retrieval.query_rewriter import LLMQueryRewriter
from retrieval.fusion import ReciprocalRankFusion
from rerank.cross_encoder import TransformerV1Reranker
from src.inference import LegalSearchEngine  # Original V1 Fallback

logger = logging.getLogger("LegalRAG.Retrieval.QueryEngine")


class Phase3QueryEngine:
    """End-to-End Phase 3 Hybrid Retrieval Engine with Stage Timing and V1 Fallback."""

    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            config = load_config("config.yaml")
        self.config = config

        self.router = CitationRouter()
        self.rewriter = LLMQueryRewriter(config)
        self.dense_indexer = DenseIndexer(config)
        self.bm25_indexer = BM25Indexer(config)
        self.rrf_fusion = ReciprocalRankFusion(rrf_k=config.get("query_pipeline", {}).get("rrf_k", 60))
        
        # Load Transformer V1 Reranker
        try:
            self.v1_reranker = TransformerV1Reranker(config)
        except Exception as e:
            logger.warning(f"Could not initialize Transformer V1 Reranker directly: {e}")
            self.v1_reranker = None

        # Load Original V1 Fallback Engine
        try:
            self.fallback_engine = LegalSearchEngine(candidate_pool_size=50)
        except Exception as e:
            logger.warning(f"Original V1 fallback engine initialization warning: {e}")
            self.fallback_engine = None

    def search(self, query_text: str) -> Dict[str, Any]:
        start_total = time.time()
        latencies = {}
        used_fallback = False
        fallback_reason = None

        try:
            # Stage 1: Citation Router
            t0 = time.time()
            route_res = self.router.route_query(query_text)
            latencies["router_time"] = round(time.time() - t0, 4)

            # Stage 2: Query Rewriting
            t1 = time.time()
            rewrite_res = self.rewriter.rewrite_query(query_text, skip_rewriting=route_res["skip_rewriting"])
            latencies["rewrite_time"] = round(time.time() - t1, 4)

            search_query = rewrite_res["rewritten_query"]

            # Stage 3a: BM25 Lexical Retrieval (Top 50)
            t2 = time.time()
            bm25_candidates = self.bm25_indexer.search(search_query, top_k=50)
            latencies["bm25_time"] = round(time.time() - t2, 4)

            # Stage 3b: Dense Semantic Retrieval (Top 50)
            t3 = time.time()
            dense_candidates = self.dense_indexer.search(search_query, top_k=50)
            latencies["dense_time"] = round(time.time() - t3, 4)

            # Stage 4: Reciprocal Rank Fusion (RRF k=60 -> Top 50)
            t4 = time.time()
            fused_candidates = self.rrf_fusion.fuse_rankings(bm25_candidates, dense_candidates, top_k=50)
            latencies["rrf_time"] = round(time.time() - t4, 4)

            # Stage 5: Transformer V1 Cross-Encoder Reranking with Case MaxP Aggregation -> Top 5 Unique Cases
            t5 = time.time()
            if self.v1_reranker:
                final_top_5, has_duplicates, duplicates_removed_count = self.v1_reranker.rerank_candidates(query_text, fused_candidates, top_k=5)
            else:
                final_top_5, has_duplicates, duplicates_removed_count = fused_candidates[:5], False, 0
            latencies["reranker_time"] = round(time.time() - t5, 4)

            latencies["total_time"] = round(time.time() - start_total, 4)

            return {
                "original_query": query_text,
                "rewritten_query": search_query,
                "was_rewritten": rewrite_res["was_rewritten"],
                "rewrite_reason": rewrite_res["reason"],
                "is_citation": route_res["is_citation"],
                "bm25_candidates": bm25_candidates[:5],
                "dense_candidates": dense_candidates[:5],
                "rrf_candidates": fused_candidates[:5],
                "final_top_5": final_top_5,
                "has_duplicate_cases_removed": has_duplicates,
                "duplicates_removed_count": duplicates_removed_count,
                "latencies": latencies,
                "used_fallback": False,
                "fallback_reason": None
            }

        except Exception as e:
            logger.error(f"Phase 3 Pipeline error ('{e}'). Triggering Fallback to Original V1 Pipeline...", exc_info=True)
            used_fallback = True
            fallback_reason = str(e)

            # Fallback Execution via LegalSearchEngine (V1)
            t_fb = time.time()
            if self.fallback_engine:
                fb_res = self.fallback_engine.search(query_text, top_k=5)
                latencies["total_time"] = round(time.time() - t_fb, 4)

                formatted_fb = []
                for r in fb_res:
                    formatted_fb.append(({
                        "chunk_id": f"{r['case_id']}_chk_0000",
                        "case_id": r['case_id'],
                        "case_name": r.get('case_name', 'Unknown'),
                        "text": r['snippet']
                    }, r['relevance_score']))

                return {
                    "original_query": query_text,
                    "rewritten_query": query_text,
                    "was_rewritten": False,
                    "rewrite_reason": "fallback_triggered",
                    "is_citation": False,
                    "bm25_candidates": [],
                    "dense_candidates": [],
                    "rrf_candidates": [],
                    "final_top_5": formatted_fb,
                    "latencies": latencies,
                    "used_fallback": True,
                    "fallback_reason": fallback_reason
                }
            else:
                raise RuntimeError(f"Both Phase 3 pipeline and Original V1 fallback failed: {e}")

