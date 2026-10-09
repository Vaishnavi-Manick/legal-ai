import os
import sys
import json
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer, AutoModel

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def mean_pooling(model_output, attention_mask):
    """Mean pooling to extract sentence/document embeddings from Transformer output."""
    token_embeddings = model_output[0]  # First element contains token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class DenseRetriever:
    """
    Dense semantic candidate retriever for AILA legal case documents
    using pretrained 'sentence-transformers/all-MiniLM-L6-v2'.
    
    Caches document embeddings on disk under models/dense_retrieval/ to avoid
    re-generating embeddings on every initialization or query.
    """
    def __init__(
        self,
        corpus_dict: Dict[str, str],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.doc_ids = list(corpus_dict.keys())
        self.num_docs = len(self.doc_ids)

        if embedding_dir is None:
            embedding_dir = os.path.join(PROJECT_ROOT, "models", "dense_retrieval")
        self.embedding_dir = embedding_dir
        os.makedirs(self.embedding_dir, exist_ok=True)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[DenseRetriever] Active compute device: {self.device}")

        # Load Tokenizer & Transformer Encoder Model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        # Cache file paths
        self.embeddings_path = os.path.join(self.embedding_dir, "doc_embeddings.pt")
        self.doc_ids_path = os.path.join(self.embedding_dir, "doc_ids.json")

        # Load or compute document embeddings
        self.doc_embeddings = self._get_or_create_embeddings(corpus_dict)

    def _get_or_create_embeddings(self, corpus_dict: Dict[str, str]) -> torch.Tensor:
        """Loads cached document embeddings if valid, else encodes all corpus docs and saves cache."""
        if os.path.exists(self.embeddings_path) and os.path.exists(self.doc_ids_path):
            try:
                with open(self.doc_ids_path, "r", encoding="utf-8") as f:
                    cached_doc_ids = json.load(f)

                if cached_doc_ids == self.doc_ids:
                    print(f"[DenseRetriever] Loading cached document embeddings from '{self.embeddings_path}'...")
                    embeddings = torch.load(self.embeddings_path, map_location=self.device)
                    print(f"[DenseRetriever] Loaded embeddings matrix shape: {embeddings.shape}")
                    return embeddings
            except Exception as e:
                print(f"[DenseRetriever] Failed to load embedding cache ({str(e)}). Re-generating embeddings...")

        print(f"[DenseRetriever] Encoding {self.num_docs} AILA documents using {self.model_name}...")
        embeddings_list = []
        batch_size = 32

        with torch.no_grad():
            for i in range(0, self.num_docs, batch_size):
                batch_doc_ids = self.doc_ids[i : i + batch_size]
                # Extract first 256 tokens / text preview per document for dense encoding
                batch_texts = [corpus_dict[doc_id][:1500] for doc_id in batch_doc_ids]

                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt"
                ).to(self.device)

                outputs = self.model(**encoded)
                embeddings = mean_pooling(outputs, encoded["attention_mask"])
                # L2 normalize embeddings for cosine similarity via dot product
                normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
                embeddings_list.append(normalized_embeddings.cpu())

        all_embeddings = torch.cat(embeddings_list, dim=0).to(self.device)

        # Save cache to disk
        print(f"[DenseRetriever] Saving precomputed embeddings to '{self.embeddings_path}'...")
        torch.save(all_embeddings.cpu(), self.embeddings_path)
        with open(self.doc_ids_path, "w", encoding="utf-8") as f:
            json.dump(self.doc_ids, f, indent=2)

        return all_embeddings

    def rank_documents(self, query_text: str, top_k: int = 50) -> List[Tuple[str, float]]:
        """
        Ranks all corpus documents against query using cosine similarity over dense embeddings.
        Returns list of (doc_id, score) tuples sorted descending by score.
        """
        if not query_text or not query_text.strip():
            return []

        with torch.no_grad():
            encoded_query = self.tokenizer(
                [query_text.strip()],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)

            query_outputs = self.model(**encoded_query)
            query_embed = mean_pooling(query_outputs, encoded_query["attention_mask"])
            normalized_query_embed = F.normalize(query_embed, p=2, dim=1)

            # Cosine similarity matrix multiplication: [1, dim] x [dim, num_docs] -> [num_docs]
            scores = torch.matmul(normalized_query_embed, self.doc_embeddings.T).squeeze(0)
            scores = scores.cpu().numpy()

        ranked_indices = scores.argsort()[::-1]
        top_k = min(top_k, self.num_docs)

        return [(self.doc_ids[idx], float(scores[idx])) for idx in ranked_indices[:top_k]]

