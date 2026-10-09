import os
import sys
import time
import logging
import torch
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer

# Ensure project root directory is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import AILADataLoader
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.transformer_dataset import chunk_document_text
from src.models.transformer import LegalTransformerCrossEncoder

logger = logging.getLogger("HybridLegalSearchEngine")
logging.basicConfig(level=logging.INFO)


class HybridLegalSearchEngine:
    """
    Hybrid Legal Case Search Engine (Phase 1 Upgrade).
    
    Stage 1: Hybrid Candidate Generation via BM25 (Top 50) + Dense MiniLM (Top 50) fused via RRF (k=60).
    Stage 2: Re-ranking via Pretrained Transformer V1 Cross-Encoder with Chunk Max-Pooling.
    
    Preserves original Transformer V1 model checkpoint and architecture without modification.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        model_dir: Optional[str] = None,
        rrf_k: int = 60,
        candidate_pool_size: int = 50,
        device: Optional[str] = None
    ):
        if data_dir is None:
            data_dir = os.path.join(PROJECT_ROOT, "data", "raw", "AILA2019")
        if model_dir is None:
            model_dir = os.path.join(PROJECT_ROOT, "models", "transformer")

        self.candidate_pool_size = candidate_pool_size
        self.rrf_k = rrf_k

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[HybridLegalSearchEngine] Active compute device: {self.device}")

        # 1. Load AILA Dataset
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Dataset directory not found: '{data_dir}'")

        self.loader = AILADataLoader(data_dir=data_dir)
        self.casedocs = self.loader.load_case_docs()
        print(f"[HybridLegalSearchEngine] Loaded {len(self.casedocs)} AILA case documents.")

        # 2. Initialize BM25 & Dense Retrievers and RRF Hybrid Retriever
        print("[HybridLegalSearchEngine] Initializing BM25 Retriever...")
        self.bm25_retriever = BM25Retriever(self.casedocs)

        print("[HybridLegalSearchEngine] Initializing Dense MiniLM-L6-v2 Retriever...")
        self.dense_retriever = DenseRetriever(self.casedocs, device=str(self.device))

        print(f"[HybridLegalSearchEngine] Initializing Hybrid RRF Retriever (k={self.rrf_k})...")
        self.hybrid_retriever = HybridRetriever(
            bm25_retriever=self.bm25_retriever,
            dense_retriever=self.dense_retriever,
            rrf_k=self.rrf_k,
            bm25_top_k=self.candidate_pool_size,
            dense_top_k=self.candidate_pool_size
        )

        # 3. Load Saved Transformer V1 Checkpoint & Tokenizer
        print(f"[HybridLegalSearchEngine] Loading Transformer V1 checkpoint & tokenizer from '{model_dir}'...")
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: '{model_dir}'")

        checkpoint_path = os.path.join(model_dir, "best_checkpoint.pt")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(model_dir, "transformer_model.pt")

        if os.path.exists(os.path.join(model_dir, "tokenizer_config.json")):
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "config" in checkpoint:
            model_name = checkpoint["config"].get("MODEL_NAME", model_name)

        self.model = LegalTransformerCrossEncoder(model_name=model_name).to(self.device)
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"[HybridLegalSearchEngine] Loaded Transformer V1 checkpoint successfully from '{checkpoint_path}'.")

    def _extract_snippet(self, text: str, max_length: int = 300) -> str:
        """Extracts clean snippet from document text."""
        cleaned = " ".join(text.split())
        if len(cleaned) <= max_length:
            return cleaned
        return cleaned[:max_length].rsplit(" ", 1)[0] + "..."

    def search_with_timing(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Executes end-to-end Hybrid Search pipeline with precise stage timing breakdown:
        1. BM25 Retrieval
        2. Dense Retrieval
        3. RRF Fusion
        4. Transformer V1 Re-ranking
        """
        if not query_text or not query_text.strip():
            return {
                "query": query_text,
                "timings": {
                    "bm25_time": 0.0,
                    "dense_time": 0.0,
                    "rrf_time": 0.0,
                    "rerank_time": 0.0,
                    "total_time": 0.0
                },
                "results": []
            }

        query_text = query_text.strip()
        start_total = time.time()

        # Step 1: BM25 Retrieval
        start_bm25 = time.time()
        bm25_results = self.bm25_retriever.rank_documents(query_text, top_k=self.candidate_pool_size)
        bm25_time = time.time() - start_bm25

        # Step 2: Dense Retrieval
        start_dense = time.time()
        dense_results = self.dense_retriever.rank_documents(query_text, top_k=self.candidate_pool_size)
        dense_time = time.time() - start_dense

        # Step 3: RRF Fusion
        start_rrf = time.time()
        rrf_scores: Dict[str, float] = {}
        for rank, (doc_id, _) in enumerate(bm25_results, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (self.rrf_k + rank))
        for rank, (doc_id, _) in enumerate(dense_results, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (self.rrf_k + rank))

        hybrid_candidates = list(rrf_scores.items())
        hybrid_candidates.sort(key=lambda x: x[1], reverse=True)
        candidate_doc_ids = [doc_id for doc_id, _ in hybrid_candidates[:self.candidate_pool_size]]
        rrf_time = time.time() - start_rrf

        # Step 4: Stage-2 Transformer V1 Re-ranking
        start_rerank = time.time()
        batch_queries = []
        batch_chunks = []
        batch_counts = []
        valid_doc_ids = []

        for doc_id in candidate_doc_ids:
            if doc_id not in self.casedocs:
                continue
            doc_text = self.casedocs[doc_id]
            chunks = chunk_document_text(
                doc_text,
                tokenizer=self.tokenizer,
                max_chunk_len=128,
                overlap=32,
                max_chunks=2
            )
            if not chunks:
                continue

            valid_doc_ids.append(doc_id)
            batch_counts.append(len(chunks))
            for c in chunks:
                batch_queries.append(query_text)
                batch_chunks.append(c)

        doc_scores = []
        if valid_doc_ids:
            eval_batch_size = 16
            with torch.no_grad():
                offset_doc = 0
                offset_chunk = 0

                while offset_doc < len(valid_doc_ids):
                    sub_counts = batch_counts[offset_doc : offset_doc + eval_batch_size]
                    sub_num_chunks = sum(sub_counts)

                    sub_queries = batch_queries[offset_chunk : offset_chunk + sub_num_chunks]
                    sub_chunks = batch_chunks[offset_chunk : offset_chunk + sub_num_chunks]

                    encoded = self.tokenizer(
                        sub_queries,
                        sub_chunks,
                        padding=True,
                        truncation=True,
                        max_length=192,
                        return_tensors="pt"
                    )

                    b_ids = encoded["input_ids"].to(self.device)
                    b_mask = encoded["attention_mask"].to(self.device)
                    b_type = encoded.get("token_type_ids", None)
                    if b_type is not None:
                        b_type = b_type.to(self.device)

                    logits = self.model(b_ids, b_mask, token_type_ids=b_type, chunk_counts=sub_counts)
                    probs = torch.sigmoid(logits).cpu().numpy().tolist()

                    if isinstance(probs, float):
                        probs = [probs]

                    doc_scores.extend(probs)
                    offset_doc += len(sub_counts)
                    offset_chunk += sub_num_chunks

        rerank_time = time.time() - start_rerank
        total_time = time.time() - start_total

        # Rank Top-K Results
        doc_score_pairs = list(zip(valid_doc_ids, doc_scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        top_results = doc_score_pairs[:top_k]

        results = []
        for rank, (doc_id, score) in enumerate(top_results, start=1):
            full_text = self.casedocs[doc_id]
            snippet = self._extract_snippet(full_text)
            results.append({
                "rank": rank,
                "case_id": doc_id,
                "relevance_score": round(float(score), 4),
                "case_name": None,
                "snippet": snippet,
                "document_text": full_text
            })

        logger.info(
            f"[Timing Breakdown] Total: {total_time:.3f}s | "
            f"BM25: {bm25_time:.3f}s | Dense: {dense_time:.3f}s | "
            f"RRF: {rrf_time:.3f}s | Transformer V1 Rerank: {rerank_time:.3f}s"
        )

        return {
            "query": query_text,
            "timings": {
                "bm25_time": round(bm25_time, 4),
                "dense_time": round(dense_time, 4),
                "rrf_time": round(rrf_time, 4),
                "rerank_time": round(rerank_time, 4),
                "total_time": round(total_time, 4)
            },
            "results": results
        }

    def search(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Standard search interface returning list of result dicts."""
        response = self.search_with_timing(query_text, top_k=top_k)
        return response["results"]

