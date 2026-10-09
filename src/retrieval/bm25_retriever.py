import math
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import Counter
from src.preprocessing import clean_text


class BM25Retriever:
    """
    Okapi BM25 candidate retriever for AILA 2019 legal case documents.
    Formula:
        Score(D, Q) = sum_{w in Q} IDF(w) * [ (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (|D| / avgdl))) ]
    """
    def __init__(
        self,
        corpus_dict: Dict[str, str],
        k1: float = 1.5,
        b: float = 0.75
    ):
        self.k1 = k1
        self.b = b
        self.doc_ids = list(corpus_dict.keys())
        self.num_docs = len(self.doc_ids)

        self.doc_tokens: List[List[str]] = []
        self.doc_lengths: List[int] = []
        self.doc_freqs: Dict[str, int] = Counter()
        self.doc_term_counts: List[Counter] = []

        total_length = 0
        for doc_id in self.doc_ids:
            tokens = clean_text(corpus_dict[doc_id])
            self.doc_tokens.append(tokens)
            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            total_length += doc_len

            term_counts = Counter(tokens)
            self.doc_term_counts.append(term_counts)

            for term in term_counts.keys():
                self.doc_freqs[term] += 1

        self.avgdl = total_length / self.num_docs if self.num_docs > 0 else 1.0

        # Precompute IDF for all terms in AILA 2019 corpus
        self.idf: Dict[str, float] = {}
        for term, df in self.doc_freqs.items():
            self.idf[term] = math.log(((self.num_docs - df + 0.5) / (df + 0.5)) + 1.0)

    def search(self, query_text: str, top_k: int = 50) -> List[Dict[str, Any]]:
        """
        Ranks all corpus documents against query using BM25 scoring.
        Returns formatted list of dicts: [{'case_id': ..., 'rank': ..., 'score': ...}]
        """
        ranked_tuples = self.rank_documents(query_text, top_k=top_k)
        results = []
        for rank, (doc_id, score) in enumerate(ranked_tuples, start=1):
            results.append({
                "case_id": doc_id,
                "rank": rank,
                "score": round(float(score), 4)
            })
        return results

    def rank_documents(self, query_text: str, top_k: int = 50) -> List[Tuple[str, float]]:
        """
        Ranks all corpus documents against query using BM25 scoring.
        Returns list of (doc_id, bm25_score) tuples sorted descending by score.
        """
        query_tokens = clean_text(query_text)
        if not query_tokens:
            return []

        scores = np.zeros(self.num_docs, dtype=np.float32)

        for token in query_tokens:
            if token not in self.idf:
                continue

            q_idf = self.idf[token]

            for i in range(self.num_docs):
                tf = self.doc_term_counts[i].get(token, 0)
                if tf == 0:
                    continue

                doc_len = self.doc_lengths[i]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                scores[i] += q_idf * (numerator / denominator)

        ranked_indices = np.argsort(-scores)
        top_k = min(top_k, self.num_docs)
        
        return [(self.doc_ids[idx], float(scores[idx])) for idx in ranked_indices[:top_k]]
