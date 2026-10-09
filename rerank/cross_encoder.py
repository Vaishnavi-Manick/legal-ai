import os
import sys
import logging
import torch
from typing import Dict, Any, List, Tuple
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.transformer import LegalTransformerCrossEncoder

logger = logging.getLogger("LegalRAG.Rerank.CrossEncoder")


class TransformerV1Reranker:
    """Wrapper for verified Transformer V1 Cross-Encoder model checkpoint with Case-Level MaxP Aggregation."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        env_cfg = config.get("environment", {})
        device_name = env_cfg.get("device", "cpu")
        if device_name == "cuda" and not torch.cuda.is_available():
            device_name = "cpu"
        self.device = torch.device(device_name)

        self.model_dir = os.path.join(PROJECT_ROOT, "models", "transformer")
        checkpoint_path = os.path.join(self.model_dir, "best_checkpoint.pt")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(self.model_dir, "transformer_model.pt")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Transformer V1 checkpoint not found at '{checkpoint_path}'.")

        logger.info(f"Loading Transformer V1 Cross-Encoder checkpoint from '{checkpoint_path}' on '{self.device}'...")
        tokenizer_src = self.model_dir if os.path.exists(os.path.join(self.model_dir, "tokenizer_config.json")) else "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        if isinstance(checkpoint, dict) and "config" in checkpoint:
            model_name = checkpoint["config"].get("MODEL_NAME", model_name)

        self.model = LegalTransformerCrossEncoder(model_name=model_name).to(self.device)
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval()
        logger.info("Transformer V1 Cross-Encoder loaded successfully.")

    def score_chunks(
        self,
        query: str,
        fused_candidates: List[Tuple[Dict[str, Any], float]]
    ) -> List[Tuple[Dict[str, Any], float]]:
        if not fused_candidates:
            return []

        queries = [query] * len(fused_candidates)
        chunk_texts = [c[0]["text"][:512] for c in fused_candidates]

        with torch.no_grad():
            encoded = self.tokenizer(
                queries,
                chunk_texts,
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

            logits = self.model(b_ids, b_mask, token_type_ids=b_type)
            probs = torch.sigmoid(logits).cpu().numpy().tolist()

            if isinstance(probs, float):
                probs = [probs]

        scored = []
        for (chunk, _), score in zip(fused_candidates, probs):
            scored.append((chunk, float(score)))

        return scored

    def rerank_candidates(
        self,
        query: str,
        fused_candidates: List[Tuple[Dict[str, Any], float]],
        top_k: int = 5
    ) -> Tuple[List[Tuple[Dict[str, Any], float]], bool, int]:
        """
        Reranks chunks by combining Transformer V1 Cross-Encoder score with Normalized RRF Retrieval score,
        then aggregates by Case ID (MaxP).
        Returns: (top_k_unique_case_results, duplicates_removed_flag, total_duplicates_removed)
        """
        scored_chunks = self.score_chunks(query, fused_candidates)
        if not scored_chunks:
            return [], False, 0

        # Min-max normalize RRF retrieval scores across the candidate pool
        rrf_scores = [c[1] for c in fused_candidates]
        min_rrf, max_rrf = min(rrf_scores), max(rrf_scores)
        rrf_range = max_rrf - min_rrf if max_rrf > min_rrf else 1.0

        # Combine Transformer V1 score with normalized RRF score
        combined_chunks = []
        for (chunk, rrf_score), (_, v1_score) in zip(fused_candidates, scored_chunks):
            norm_rrf = (rrf_score - min_rrf) / rrf_range
            hybrid_score = v1_score + 0.05 * norm_rrf
            combined_chunks.append((chunk, hybrid_score))

        # Case-level MaxP aggregation using hybrid score
        best_chunk_per_case: Dict[str, Dict[str, Any]] = {}
        best_score_per_case: Dict[str, float] = {}

        total_chunks = len(combined_chunks)

        for chunk, score in combined_chunks:
            case_id = chunk["case_id"]
            if case_id not in best_score_per_case or score > best_score_per_case[case_id]:
                best_score_per_case[case_id] = score
                best_chunk_per_case[case_id] = chunk

        unique_cases_count = len(best_score_per_case)
        duplicates_removed = total_chunks - unique_cases_count
        has_duplicates = duplicates_removed > 0

        # Build unique case rankings
        unique_rankings = []
        for case_id, score in best_score_per_case.items():
            unique_rankings.append((best_chunk_per_case[case_id], score))

        # Sort by best chunk score descending
        unique_rankings.sort(key=lambda x: x[1], reverse=True)

        logger.info(f"Case Aggregation (MaxP + RRF Fusion): {total_chunks} chunks -> {unique_cases_count} unique cases ({duplicates_removed} duplicate case chunks removed).")
        return unique_rankings[:top_k], has_duplicates, duplicates_removed

