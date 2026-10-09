import os
import json
import logging
import torch
import torch.nn.functional as F
from datetime import datetime
from typing import Dict, Any, List, Tuple
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger("LegalRAG.Index.DenseIndexer")


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class DenseIndexer:
    """Dense Semantic Indexer with batch checkpointing and device configuration."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        env_cfg = config.get("environment", {})
        paths_cfg = config.get("paths", {})

        device_name = env_cfg.get("device", "cpu")
        if device_name == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device_name = "cpu"

        self.device = torch.device(device_name)
        self.model_name = env_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        self.output_dir = os.path.abspath(paths_cfg.get("vector_db_dir", "data/dense_index"))

        logger.info(f"Initializing DenseIndexer on device '{self.device}' using model '{self.model_name}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        self.embedding_dim = getattr(self.model.config, "hidden_size", 384)

    def build_index_with_checkpoint(self, chunks: List[Dict[str, Any]], batch_size: int = 32) -> Dict[str, Any]:
        os.makedirs(self.output_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.output_dir, "checkpoint.json")
        embeddings_path = os.path.join(self.output_dir, "chunk_embeddings.pt")
        metadata_path = os.path.join(self.output_dir, "chunks_metadata.json")
        index_meta_path = os.path.join(self.output_dir, "index_metadata.json")

        # Load existing checkpoint if present
        processed_chunk_ids = set()
        existing_embeddings_list = []
        existing_chunks_meta = []

        if os.path.exists(checkpoint_path) and os.path.exists(embeddings_path) and os.path.exists(metadata_path):
            try:
                with open(checkpoint_path, "r", encoding="utf-8") as f:
                    ckpt = json.load(f)
                    processed_chunk_ids = set(ckpt.get("processed_chunk_ids", []))

                with open(metadata_path, "r", encoding="utf-8") as f:
                    existing_chunks_meta = json.load(f)

                existing_embeddings = torch.load(embeddings_path, map_location="cpu")
                existing_embeddings_list.append(existing_embeddings)
                logger.info(f"Resuming indexing from checkpoint: {len(processed_chunk_ids)} chunks already processed.")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint, starting fresh: {e}")
                processed_chunk_ids = set()

        unprocessed_chunks = [c for c in chunks if c["chunk_id"] not in processed_chunk_ids]
        logger.info(f"Total chunks: {len(chunks)} | Already indexed: {len(processed_chunk_ids)} | Remaining to index: {len(unprocessed_chunks)}")

        new_embeddings = []
        new_chunks_meta = []

        with torch.no_grad():
            for i in range(0, len(unprocessed_chunks), batch_size):
                batch_chunks = unprocessed_chunks[i : i + batch_size]
                texts = [c["text"][:1500] for c in batch_chunks]

                encoded = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt"
                ).to(self.device)

                outputs = self.model(**encoded)
                embeds = mean_pooling(outputs, encoded["attention_mask"])
                norm_embeds = F.normalize(embeds, p=2, dim=1).cpu()

                new_embeddings.append(norm_embeds)
                for c in batch_chunks:
                    new_chunks_meta.append(c)
                    processed_chunk_ids.add(c["chunk_id"])

                # Checkpoint save after every batch
                all_current_meta = existing_chunks_meta + new_chunks_meta
                current_tensor = torch.cat(existing_embeddings_list + new_embeddings, dim=0)

                torch.save(current_tensor, embeddings_path)
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(all_current_meta, f, indent=2)

                with open(checkpoint_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "processed_chunk_ids": list(processed_chunk_ids),
                        "last_updated": datetime.now().isoformat()
                    }, f, indent=2)

        # Save index metadata
        final_chunks_meta = existing_chunks_meta + new_chunks_meta
        final_embeddings = torch.cat(existing_embeddings_list + new_embeddings, dim=0) if (existing_embeddings_list or new_embeddings) else torch.empty((0, self.embedding_dim))

        index_metadata = {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "total_chunks_indexed": len(final_chunks_meta),
            "device_used": str(self.device),
            "normalization": "L2",
            "last_updated": datetime.now().isoformat(),
            "chunk_ids": [c["chunk_id"] for c in final_chunks_meta]
        }

        with open(index_meta_path, "w", encoding="utf-8") as f:
            json.dump(index_metadata, f, indent=2)

        logger.info(f"Dense indexing complete. Index size: {final_embeddings.shape}")
        return index_metadata

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        embeddings_path = os.path.join(self.output_dir, "chunk_embeddings.pt")
        metadata_path = os.path.join(self.output_dir, "chunks_metadata.json")

        if not os.path.exists(embeddings_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError("Dense index files missing. Please run indexing first.")

        doc_embeddings = torch.load(embeddings_path, map_location=self.device)
        with open(metadata_path, "r", encoding="utf-8") as f:
            chunks_meta = json.load(f)

        with torch.no_grad():
            encoded_q = self.tokenizer([query], padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.device)
            outputs_q = self.model(**encoded_q)
            q_embed = mean_pooling(outputs_q, encoded_q["attention_mask"])
            q_norm = F.normalize(q_embed, p=2, dim=1)

            scores = torch.mm(q_norm, doc_embeddings.to(self.device).T).squeeze(0)
            top_scores, top_indices = torch.topk(scores, min(top_k, len(chunks_meta)))

        results = []
        for idx, score in zip(top_indices.cpu().numpy(), top_scores.cpu().numpy()):
            results.append((chunks_meta[int(idx)], float(score)))

        return results

