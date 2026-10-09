import os
import json
import logging
from typing import Dict, Any, List
from ingest.pipeline import load_config
from index.dense_indexer import DenseIndexer
from index.bm25_indexer import BM25Indexer

logger = logging.getLogger("LegalRAG.Index.Pipeline")


class IndexingPipeline:
    """Orchestrates Phase 2 Dense & BM25 Indexing with verification."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dense_indexer = DenseIndexer(config)
        self.bm25_indexer = BM25Indexer(config)

    def run_indexing(self, chunks_path: str = None) -> Dict[str, Any]:
        if not chunks_path:
            chunks_path = os.path.join(self.config["paths"]["processed_dir"], "chunks.json")

        full_chunks_path = os.path.abspath(chunks_path)
        if not os.path.exists(full_chunks_path):
            raise FileNotFoundError(f"Processed chunks file not found at '{full_chunks_path}'. Run Phase 1 first.")

        with open(full_chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        logger.info(f"Loaded {len(chunks)} chunks for Phase 2 Indexing.")

        # 1. Build Dense Index (with batch checkpointing)
        dense_meta = self.dense_indexer.build_index_with_checkpoint(chunks)

        # 2. Build BM25 Index (persisted)
        bm25_meta = self.bm25_indexer.build_index(chunks)

        # 3. Verification: Verify 1-to-1 mapping of chunk IDs
        dense_chunk_ids = dense_meta["chunk_ids"]
        bm25_chunk_ids = bm25_meta["chunk_ids"]

        is_aligned = (dense_chunk_ids == bm25_chunk_ids)
        if not is_aligned:
            logger.error("Mismatch detected between Dense and BM25 indexed chunk IDs!")
            raise ValueError("Chunk ID alignment failed between Dense and BM25 indexes.")

        logger.info(f"Verified 1-to-1 chunk ID alignment for all {len(dense_chunk_ids)} chunks.")

        summary = {
            "total_chunks": len(chunks),
            "indexed_successfully": len(dense_chunk_ids),
            "indexing_failures": 0,
            "alignment_verified": is_aligned,
            "dense_model": dense_meta["model_name"],
            "embedding_dim": dense_meta["embedding_dimension"],
            "bm25_algorithm": bm25_meta["algorithm"]
        }
        return summary

