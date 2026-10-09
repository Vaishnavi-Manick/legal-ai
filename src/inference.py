import os
import sys
import time
import logging
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer


logger = logging.getLogger("LegalSearchEngine")


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import AILADataLoader
from src.preprocessing import TFIDFRetriever
from src.transformer_dataset import chunk_document_text
from src.models.transformer import LegalTransformerCrossEncoder


class LegalSearchEngine:
    """
    Two-Stage Legal Case Document Search Engine for AILA 2019.
    
    Stage 1: Candidate Generation via TF-IDF Retriever over 2,914 AILA case documents.
    Stage 2: Re-ranking via Pretrained Legal Transformer Cross-Encoder with Chunk Max-Pooling.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        model_dir: Optional[str] = None,
        candidate_pool_size: int = 50,
        device: Optional[str] = None
    ):
        """
        Initializes the search engine:
        - Loads AILA case documents from disk using AILADataLoader
        - Builds Stage-1 TFIDFRetriever index
        - Loads trained Transformer Cross-Encoder & Tokenizer once into memory
        - Sets model to evaluation mode (`model.eval()`)
        """
        # 1. Resolve default paths
        if data_dir is None:
            data_dir = os.path.join(PROJECT_ROOT, "data", "raw", "AILA2019")
        if model_dir is None:
            model_dir = os.path.join(PROJECT_ROOT, "models", "transformer")

        self.candidate_pool_size = candidate_pool_size

        # 2. Configure compute device (CUDA if available, else CPU)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            if device.lower() == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but unavailable on system. Falling back to CPU.")
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(device)

        print(f"[LegalSearchEngine] Active compute device: {self.device}")

        # 3. Load AILA Case Documents with error handling
        print(f"[LegalSearchEngine] Loading dataset from '{data_dir}'...")
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Dataset directory not found: '{data_dir}'")

        try:
            self.loader = AILADataLoader(data_dir=data_dir)
            self.casedocs = self.loader.load_case_docs()
            if not self.casedocs:
                raise ValueError(f"No case document files found in '{data_dir}'")
            print(f"[LegalSearchEngine] Loaded {len(self.casedocs)} AILA case documents.")
        except Exception as e:
            raise RuntimeError(f"Error loading AILA dataset: {str(e)}") from e

        # 4. Initialize Stage-1 Candidate Retriever
        print("[LegalSearchEngine] Initializing Stage-1 TF-IDF candidate retriever...")
        try:
            self.tfidf_retriever = TFIDFRetriever(self.casedocs)
        except Exception as e:
            raise RuntimeError(f"Error initializing TFIDFRetriever: {str(e)}") from e

        # 5. Load Saved Transformer Tokenizer & Model Checkpoint
        print(f"[LegalSearchEngine] Loading Transformer checkpoint & tokenizer from '{model_dir}'...")
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: '{model_dir}'")

        checkpoint_path = os.path.join(model_dir, "best_checkpoint.pt")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(model_dir, "transformer_model.pt")

        try:
            # Load tokenizer from saved model_dir or default HuggingFace model
            if os.path.exists(os.path.join(model_dir, "tokenizer_config.json")):
                self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

            # Determine underlying model architecture name
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                if isinstance(checkpoint, dict) and "config" in checkpoint:
                    model_name = checkpoint["config"].get("MODEL_NAME", model_name)

            # Instantiate model architecture
            self.model = LegalTransformerCrossEncoder(model_name=model_name).to(self.device)

            # Load trained weights
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
                self.model.load_state_dict(state_dict)
                print(f"[LegalSearchEngine] Loaded checkpoint weights from '{checkpoint_path}'.")
            else:
                raise FileNotFoundError(f"Checkpoint file not found in '{model_dir}'")

            # Put model into evaluation mode
            self.model.eval()

        except Exception as e:
            raise RuntimeError(f"Error loading Transformer model: {str(e)}") from e

    def _extract_snippet(self, text: str, max_length: int = 300) -> str:
        """Extracts a clean snippet from the case document text."""
        cleaned = " ".join(text.split())
        if len(cleaned) <= max_length:
            return cleaned
        return cleaned[:max_length].rsplit(" ", 1)[0] + "..."

    def search(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Executes end-to-end inference pipeline:
        1. Validates query input
        2. Retrieves top-N candidate documents via Stage-1 TF-IDF
        3. Chunks documents via chunk_document_text()
        4. Cross-encoder scoring via Transformer with Max-Pooling over chunks
        5. Ranks documents by relevance score and returns top_k results
        """
        # 1. Query Validation
        if not query_text or not query_text.strip():
            print("[LegalSearchEngine] Empty query provided. Returning empty result list.")
            return []

        query_text = query_text.strip()
        top_k = max(1, top_k)

        # 2. Stage-1 Candidate Retrieval via TF-IDF
        ranked_candidates = self.tfidf_retriever.rank_documents(query_text)
        candidate_doc_ids = [doc_id for doc_id, _ in ranked_candidates[:self.candidate_pool_size]]

        if not candidate_doc_ids:
            return []

        # 3. Stage-2 Document Chunking and Pair Constructing
        batch_queries = []
        batch_chunks = []
        batch_counts = []
        valid_doc_ids = []

        for doc_id in candidate_doc_ids:
            if doc_id not in self.casedocs:
                continue
            doc_text = self.casedocs[doc_id]
            # Chunking matching training hyperparameters (max_len=128, overlap=32, max_chunks=2)
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

        if not valid_doc_ids:
            return []

        # 4. Stage-2 Transformer Scoring (torch.no_grad for inference safety & speed)
        doc_scores = []
        eval_batch_size = 16  # Sub-batch size to prevent GPU/CPU OOM

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

                # Forward pass: max-pools logits across document chunks
                logits = self.model(b_ids, b_mask, token_type_ids=b_type, chunk_counts=sub_counts)
                probs = torch.sigmoid(logits).cpu().numpy().tolist()

                if isinstance(probs, float):
                    probs = [probs]

                doc_scores.extend(probs)

                offset_doc += len(sub_counts)
                offset_chunk += sub_num_chunks

        # 5. Ranking & Constructing Top-K Result List
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
                "case_name": None,  # No case metadata in AILA 2019 dataset
                "snippet": snippet,
                "document_text": full_text
            })

        return results

