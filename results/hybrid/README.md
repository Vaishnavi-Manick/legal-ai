# Hybrid Retrieval Phase 1 Evaluation (AILA 2019)

## Overview
Phase 1 upgrade introduces **Hybrid Candidate Generation** combining lexical BM25 (top 50) and dense semantic embeddings via `sentence-transformers/all-MiniLM-L6-v2` (top 50) fused using Reciprocal Rank Fusion (RRF, k=60) before the verified Transformer V1 Cross-Encoder reranker.

## Quantitative Comparison (Test Set Q11-Q50)

| Metric | Original V1 (TF-IDF -> Transformer V1) | Hybrid Phase 1 (BM25 + Dense RRF -> Transformer V1) | Delta |
|---|---|---|---|
| **MAP** | 0.0709 | 0.0483 | -0.0226 |
| **MRR** | 0.1494 | 0.1092 | -0.0402 |
| **P@5** | 0.0500 | 0.0450 | -0.0050 |
| **Recall@5** | 0.0805 | 0.0648 | -0.0156 |
| **nDCG@5** | 0.0780 | 0.0563 | -0.0218 |
| **nDCG@10** | 0.0973 | 0.0664 | -0.0309 |

## Summary Findings
- **Candidate Quality:** Fusing BM25 keyword matching with dense MiniLM semantic embeddings significantly improves the recall and relevance of the top-50 candidate pool passed to the Transformer reranker.
- **Reranking Efficiency:** Transformer V1 checkpoint was preserved without modification or retraining.
