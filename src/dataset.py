import random
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Set
from src.preprocessing import Vocabulary

class BiLSTMPairDataset(Dataset):
    """
    Constructs (query, document, relevance_label) pairs for supervised BiLSTM relevance modeling.
    Supports positive relevance pairs (label=1.0) and negative sampling (label=0.0).
    """
    def __init__(
        self,
        qids: List[str],
        queries: Dict[str, str],
        casedocs: Dict[str, str],
        qrels: Dict[str, Set[str]],
        vocab: Vocabulary,
        num_negatives: int = 5,
        max_q_len: int = 256,
        max_d_len: int = 512
    ):
        self.samples = []
        self.vocab = vocab
        self.max_q_len = max_q_len
        self.max_d_len = max_d_len
        
        all_doc_ids = set(casedocs.keys())
        
        print(f"Constructing binary relevance pairs for {len(qids)} queries (num_negatives={num_negatives})...")
        for qid in qids:
            q_text = queries[qid]
            pos_docs = qrels.get(qid, set())
            if not pos_docs:
                continue
                
            non_relevant_docs = list(all_doc_ids - pos_docs)
            
            for pos_doc_id in pos_docs:
                if pos_doc_id not in casedocs:
                    continue
                
                # 1. Positive pair
                self.samples.append({
                    "query_id": qid,
                    "document_id": pos_doc_id,
                    "q_text": q_text,
                    "doc_text": casedocs[pos_doc_id],
                    "label": 1.0
                })
                
                # 2. Negative pairs
                sample_count = min(num_negatives, len(non_relevant_docs))
                neg_doc_ids = random.sample(non_relevant_docs, sample_count)
                for neg_doc_id in neg_doc_ids:
                    if neg_doc_id in casedocs:
                        self.samples.append({
                            "query_id": qid,
                            "document_id": neg_doc_id,
                            "q_text": q_text,
                            "doc_text": casedocs[neg_doc_id],
                            "label": 0.0
                        })

        print(f"Total training pair samples created: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        q_ids = self.vocab.text_to_ids(sample["q_text"], max_len=self.max_q_len)
        doc_ids = self.vocab.text_to_ids(sample["doc_text"], max_len=self.max_d_len)
        
        return {
            "query_id": sample["query_id"],
            "document_id": sample["document_id"],
            "q_ids": q_ids,
            "doc_ids": doc_ids,
            "label": sample["label"]
        }

def pad_sequence(seq: List[int], max_len: int, pad_value: int = 0) -> List[int]:
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [pad_value] * (max_len - len(seq))

def collate_bilstm_pairs(batch, max_q_len: int = 256, max_d_len: int = 512):
    """
    Collate function to pad variable-length sequences into uniform Tensors.
    """
    batch_q = [pad_sequence(item["q_ids"], max_q_len) for item in batch]
    batch_d = [pad_sequence(item["doc_ids"], max_d_len) for item in batch]
    batch_labels = [item["label"] for item in batch]
    
    return {
        "query_ids": torch.tensor(batch_q, dtype=torch.long),
        "doc_ids": torch.tensor(batch_d, dtype=torch.long),
        "labels": torch.tensor(batch_labels, dtype=torch.float32)
    }

