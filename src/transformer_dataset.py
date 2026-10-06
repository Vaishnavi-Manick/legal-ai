import random
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Set, Tuple

def chunk_document_text(text: str, tokenizer, max_chunk_len: int = 128, overlap: int = 32, max_chunks: int = 2) -> List[str]:
    """
    Splits long legal case documents into overlapping chunks.
    Ensures long documents are not blindly truncated while keeping memory minimal.
    """
    words = text.split()
    if len(words) <= max_chunk_len:
        return [text]
    
    chunks = []
    step = max_chunk_len - overlap
    for i in range(0, len(words), step):
        chunk_words = words[i : i + max_chunk_len]
        chunk_str = " ".join(chunk_words)
        chunks.append(chunk_str)
        if len(chunks) >= max_chunks:
            break
            
    return chunks

class LegalTransformerPairDataset(Dataset):
    """
    PyTorch Dataset for Supervised Legal Query-Document Relevance.
    Generates Query + Document Chunk pairs for Cross-Encoder training.
    """
    def __init__(
        self,
        qids: List[str],
        queries: Dict[str, str],
        casedocs: Dict[str, str],
        qrels: Dict[str, Set[str]],
        tokenizer,
        num_negatives: int = 2,
        max_q_len: int = 64,
        max_d_chunk_len: int = 128,
        max_chunks_per_doc: int = 2
    ):
        self.qids = qids
        self.queries = queries
        self.casedocs = casedocs
        self.qrels = qrels
        self.tokenizer = tokenizer
        self.num_negatives = num_negatives
        self.max_q_len = max_q_len
        self.max_d_chunk_len = max_d_chunk_len
        self.max_chunks_per_doc = max_chunks_per_doc
        self.all_doc_ids = list(casedocs.keys())
        
        self.samples = []
        self._build_samples()
        
    def _build_samples(self):
        """Constructs balanced positive and negative query-document training pairs."""
        for qid in self.qids:
            if qid not in self.queries:
                continue
            q_text = self.queries[qid]
            pos_doc_ids = list(self.qrels.get(qid, set()))
            neg_candidate_pool = list(set(self.all_doc_ids) - set(pos_doc_ids))
            
            for pos_id in pos_doc_ids:
                if pos_id not in self.casedocs:
                    continue
                # Add positive pair
                self.samples.append((qid, q_text, pos_id, self.casedocs[pos_id], 1.0))
                
                # Sample negatives for each positive pair
                sampled_negs = random.sample(neg_candidate_pool, min(self.num_negatives, len(neg_candidate_pool)))
                for neg_id in sampled_negs:
                    if neg_id in self.casedocs:
                        self.samples.append((qid, q_text, neg_id, self.casedocs[neg_id], 0.0))
                        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        qid, q_text, doc_id, doc_text, label = self.samples[idx]
        chunks = chunk_document_text(
            doc_text,
            tokenizer=self.tokenizer,
            max_chunk_len=self.max_d_chunk_len,
            overlap=32,
            max_chunks=self.max_chunks_per_doc
        )
        return {
            "query_id": qid,
            "doc_id": doc_id,
            "query_text": q_text,
            "doc_chunks": chunks,
            "label": label
        }

def collate_transformer_chunks(batch, tokenizer, max_seq_len: int = 192):
    """
    Batches query-document chunk pairs into tokenized PyTorch tensors.
    """
    flat_queries = []
    flat_chunks = []
    chunk_counts = []
    labels = []
    
    for item in batch:
        q_text = item["query_text"]
        chunks = item["doc_chunks"]
        labels.append(item["label"])
        chunk_counts.append(len(chunks))
        
        for c in chunks:
            flat_queries.append(q_text)
            flat_chunks.append(c)
            
    encoded = tokenizer(
        flat_queries,
        flat_chunks,
        padding=True,
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt"
    )
    
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "token_type_ids": encoded.get("token_type_ids", None),
        "chunk_counts": chunk_counts,
        "labels": torch.tensor(labels, dtype=torch.float32)
    }
