import os
import re
import json
import pickle
import logging
from typing import Dict, Any, List, Tuple
from rank_bm25 import BM25Okapi

logger = logging.getLogger("LegalRAG.Index.BM25Indexer")


class BM25Indexer:
    """Persisted Okapi BM25 Lexical Indexer mapping chunk_ids 1-to-1."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        paths_cfg = config.get("paths", {})
        self.output_dir = os.path.abspath(paths_cfg.get("bm25_index_dir", "data/bm25_index"))
        self.bm25_model = None
        self.chunks_meta = []

    def _tokenize(self, text: str) -> List[str]:
        # Lowercase, clean alphanumeric tokens
        tokens = re.findall(r"\w+", text.lower())
        return [t for t in tokens if len(t) > 2]

    def build_index(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        os.makedirs(self.output_dir, exist_ok=True)
        data_path = os.path.join(self.output_dir, "bm25_data.pkl")
        meta_path = os.path.join(self.output_dir, "bm25_metadata.json")

        logger.info(f"Building BM25 index over {len(chunks)} chunks...")
        self.chunks_meta = chunks
        tokenized_corpus = [self._tokenize(c["text"]) for c in chunks]

        self.bm25_model = BM25Okapi(tokenized_corpus)

        # Persist index and chunks mapping
        with open(data_path, "wb") as f:
            pickle.dump({
                "bm25_model": self.bm25_model,
                "chunks_meta": self.chunks_meta
            }, f)

        metadata = {
            "total_chunks_indexed": len(chunks),
            "algorithm": "BM25Okapi",
            "k1": 1.5,
            "b": 0.75,
            "chunk_ids": [c["chunk_id"] for c in chunks]
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved BM25 index to '{data_path}' and metadata to '{meta_path}'.")
        return metadata

    def load_index(self):
        data_path = os.path.join(self.output_dir, "bm25_data.pkl")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"BM25 index not found at '{data_path}'. Please build index first.")

        with open(data_path, "rb") as f:
            data = pickle.load(f)
            self.bm25_model = data["bm25_model"]
            self.chunks_meta = data["chunks_meta"]

        logger.info(f"Loaded BM25 index with {len(self.chunks_meta)} chunks.")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        if self.bm25_model is None:
            self.load_index()

        tokenized_query = self._tokenize(query)
        scores = self.bm25_model.get_scores(tokenized_query)

        # Pair scores with chunk metadata
        paired = list(zip(self.chunks_meta, scores))
        paired.sort(key=lambda x: x[1], reverse=True)

        return [(chunk, float(score)) for chunk, score in paired[:top_k]]

