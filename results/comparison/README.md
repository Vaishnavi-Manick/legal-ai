# Model Evaluation & Selection Summary

## 📌 Overview
This document summarizes the comparative empirical evaluation of legal precedent retrieval architectures trained and evaluated on the **AILA 2019 Legal Case Retrieval dataset**.

---

## 🔬 Experimental Setup & Evaluation Methodology
- **Dataset**: AILA 2019 Case Retrieval Corpus (2,914 legal case documents).
- **Test Set**: 40 Test Queries (`Q11`–`Q50`) with official relevance judgments (`qrels`).
- **Data Split Integrity**: Identical train/val/test query split protocol across all models.
- **Evaluation Metrics**: Standard TREC Information Retrieval (IR) metrics:
  - **P@5** (Precision at rank 5)
  - **Recall@5** (Recall at rank 5)
  - **MRR** (Mean Reciprocal Rank)
  - **MAP** (Mean Average Precision)
  - **NDCG@5 & NDCG@10** (Normalized Discounted Cumulative Gain at ranks 5 & 10)

---

## 📊 Final Comparative Results

| Model Architecture | Precision@5 | Recall@5 | MRR | MAP | NDCG@5 | NDCG@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Siamese BiLSTM + Attention** | 0.0000 | 0.0000 | 0.0082 | 0.0068 | 0.0000 | 0.0051 |
| **Legal Transformer Cross-Encoder** | **0.1250** | **0.1881** | **0.2790** | **0.1886** | **0.1774** | **0.2008** |

---

## 💡 Findings & Model Selection Rationale

1. **Performance Breakdown**:
   - **Siamese BiLSTM + Attention**: Struggles to capture complex legal semantics and long-range clause context over length-limited sequences, achieving a low MAP of `0.0068` and MRR of `0.0082`.
   - **Legal Transformer Cross-Encoder**: Substantially outperforms the BiLSTM model across every single IR evaluation metric, reaching a MAP of `0.1886` (+26.7x improvement) and MRR of `0.2790` (+33.0x improvement).

2. **Final Model Selection**:
   - The **Legal Transformer Cross-Encoder** (`sentence-transformers/all-MiniLM-L6-v2` with sliding-window chunking and Max-Pooling aggregation) has been selected as the **final production retrieval engine** for the Legal AI search system and Streamlit frontend interface.
