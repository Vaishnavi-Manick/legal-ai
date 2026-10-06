import re
import math
import numpy as np
from typing import Dict, List, Tuple, Set
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def clean_text(text: str) -> List[str]:
    """Tokenizes and cleans text into lowercase alpha-numeric tokens."""
    tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
    return tokens

class Vocabulary:
    def __init__(self, pad_token="<pad>", unk_token="<unk>"):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.w2i = {pad_token: 0, unk_token: 1}
        self.i2w = {0: pad_token, 1: unk_token}

    def build_vocab(self, texts: List[str], max_vocab_size: int = 30000, min_freq: int = 2):
        """Fit vocabulary strictly on provided training texts to prevent data leakage."""
        counter = Counter()
        for text in texts:
            counter.update(clean_text(text))
        
        most_common = counter.most_common(max_vocab_size)
        idx = 2
        for word, freq in most_common:
            if freq >= min_freq:
                self.w2i[word] = idx
                self.i2w[idx] = word
                idx += 1

    def __len__(self):
        return len(self.w2i)

    def text_to_ids(self, text: str, max_len: int = 512) -> List[int]:
        tokens = clean_text(text)[:max_len]
        ids = [self.w2i.get(token, self.w2i[self.unk_token]) for token in tokens]
        if not ids:
            ids = [self.w2i[self.pad_token]]
        return ids

def compute_lsa_embeddings(corpus_texts: List[str], vocab: Vocabulary, embedding_dim: int = 128) -> np.ndarray:
    """
    Computes deterministic, corpus-only LSA word embeddings via TruncatedSVD on TF-IDF word-document matrix.
    Returns embedding matrix of shape (vocab_size, embedding_dim).
    """
    print(f"Building LSA embeddings (dim={embedding_dim}) for vocabulary size={len(vocab)}...")
    vocab_words = [vocab.i2w[i] for i in range(2, len(vocab))]
    word_to_col = {word: i for i, word in enumerate(vocab_words)}
    
    vectorizer = TfidfVectorizer(vocabulary=word_to_col, norm='l2', sublinear_tf=True)
    doc_term_matrix = vectorizer.fit_transform(corpus_texts)
    term_doc_matrix = doc_term_matrix.T
    
    n_components = min(embedding_dim, term_doc_matrix.shape[1] - 1, term_doc_matrix.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    term_embeddings = svd.fit_transform(term_doc_matrix)
    
    if term_embeddings.shape[1] < embedding_dim:
        pad_cols = np.zeros((term_embeddings.shape[0], embedding_dim - term_embeddings.shape[1]))
        term_embeddings = np.hstack([term_embeddings, pad_cols])
        
    full_embed_matrix = np.zeros((len(vocab), embedding_dim), dtype=np.float32)
    full_embed_matrix[2:] = term_embeddings
    if len(term_embeddings) > 0:
        full_embed_matrix[1] = np.mean(term_embeddings, axis=0)
    
    norms = np.linalg.norm(full_embed_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    full_embed_matrix = full_embed_matrix / norms
    
    return full_embed_matrix

class TFIDFRetriever:
    """TF-IDF retriever using scikit-learn for candidate ranking."""
    def __init__(self, corpus_dict: Dict[str, str]):
        self.doc_ids = list(corpus_dict.keys())
        self.vectorizer = TfidfVectorizer(norm='l2', sublinear_tf=True, stop_words='english')
        corpus_texts = [corpus_dict[doc_id] for doc_id in self.doc_ids]
        self.doc_matrix = self.vectorizer.fit_transform(corpus_texts)

    def rank_documents(self, query_text: str) -> List[Tuple[str, float]]:
        q_vec = self.vectorizer.transform([query_text])
        scores = (self.doc_matrix * q_vec.T).toarray().ravel()
        ranked_indices = np.argsort(-scores)
        return [(self.doc_ids[idx], float(scores[idx])) for idx in ranked_indices]


