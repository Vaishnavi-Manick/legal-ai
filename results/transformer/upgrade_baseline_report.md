# Baseline Architecture & Data Pipeline Inspection Report

## 📌 Executive Summary
This report documents the deep inspection of the existing **Legal Transformer Cross-Encoder** baseline codebase (`Legal AI` project) prior to Phase 2–11 upgrades. The inspection verified dataset loading, train/validation/test query splits, qrels ground truth handling, candidate retrieval, document chunking, and metric calculation logic.

---

## 🔍 Detailed Component Inspection

### 1. Dataset & Qrels Parsing (`src/data_loader.py`)
- **Query File**: `data/raw/AILA2019/Query_doc.txt` (50 queries, `AILA_Q1` to `AILA_Q50`).
- **Case Documents**: `data/raw/AILA2019/Object_casedocs/*.txt` (2,914 individual case text files).
- **Relevance Judgments (`qrels`)**: `data/raw/AILA2019/relevance_judgments_priorcases.txt` in TREC format (`qid 0 doc_id rel`). Parsed into `qrels[qid] = {doc_ids}` where `rel > 0`.

### 2. Query Splits & Data Leakage Audit (`src/data_loader.py`)
- **Train Split**: Queries `Q1`–`Q8` (8 queries with positive ground truth).
- **Validation Split**: Queries `Q9`–`Q10` (2 queries with positive ground truth).
- **Test Split**: Queries `Q11`–`Q50` (40 queries with positive ground truth).
- **Data Leakage Audit**: **PASSED CLEANLY**.
  - Test queries `Q11`–`Q50` are strictly excluded from dataset sample generation during training.
  - The model is trained purely on `Q1`–`Q8` and evaluated zero-shot on `Q11`–`Q50`.

### 3. Training Pair Construction (`src/transformer_dataset.py`)
- **Positive Pairs**: For each train query `qid`, every known relevant case `doc_id` in `qrels[qid]` is paired with `q_text` with label `1.0`.
- **Negative Pairs**: Currently constructed via **Random Uniform Sampling** from `set(all_doc_ids) - set(pos_doc_ids)` with `num_negatives = 2` per positive.
- **Key Observation for Upgrade**: Random negatives are often trivial to distinguish (e.g. completely unrelated legal topics), limiting the model's ability to learn fine-grained legal distinctions. Hard negative mining via TF-IDF (Phase 2) will directly address this.

### 4. Document Chunking Strategy (`src/transformer_dataset.py`)
- **Current Parameters**:
  - `max_chunk_len`: 128 words
  - `overlap`: 32 words
  - `max_chunks`: 2 chunks per document
- **Aggregation Method**: Max-Pooling ($S(Q, D) = \max_k \text{logit}(Q, D_{\text{chunk } k})$) inside `LegalTransformerCrossEncoder.forward()`.
- **Key Observation for Upgrade**: Long legal documents (~2,000 words) are truncated after ~224 words (2 chunks * 128 words minus overlap). Expanding chunk length to 256 or 384 words (Phase 3) will capture substantially more legal precedent context.

### 5. Candidate Retrieval Mechanism (`src/preprocessing.py`)
- **Class**: `TFIDFRetriever`
- **Vectorization**: `TfidfVectorizer(norm='l2', sublinear_tf=True, stop_words='english')` across 2,914 corpus documents.
- **Scoring**: Cosine similarity dot product $(D_{\text{tfidf}} \cdot Q_{\text{tfidf}}^T)$.
- **Candidate Pool**: Top 50 documents per query fed to Transformer for re-ranking.

### 6. Metric Calculation (`src/utils.py` & `src/evaluation.py`)
- Standard TREC IR Metrics computed over test set (`Q11`–`Q50`):
  - **P@5**: Precision at rank 5.
  - **Recall@5**: Proportion of relevant documents retrieved in top 5.
  - **MRR**: Mean Reciprocal Rank ($1 / \text{first\_relevant\_rank}$).
  - **MAP**: Mean Average Precision across test queries.
  - **nDCG@5 & nDCG@10**: Normalized Discounted Cumulative Gain at ranks 5 & 10.

---

## 📊 Verified Baseline Performance (Transformer V1)

$$\begin{array}{lcccccc}
\hline
\textbf{Model Version} & \textbf{P@5} & \textbf{Recall@5} & \textbf{MRR} & \textbf{MAP} & \textbf{nDCG@5} & \textbf{nDCG@10} \\
\hline
\text{Transformer Baseline (V1)} & 0.1250 & 0.1881 & 0.2790 & 0.1886 & 0.1774 & 0.2008 \\
\hline
\end{array}$$

---

## 🎯 Target Upgrades for Phase 2–11
1. **Hard Negative Mining (Phase 2)**: Replace random negatives with top TF-IDF candidate non-relevant cases to force the model to learn fine-grained legal distinctions.
2. **Chunking Experiments (Phase 3)**: Compare 128-, 256-, and 384-word chunk sizes with increased max chunks.
3. **Transformer V2 (Phase 4–7)**: Save to `models/transformer_v2/`, evaluate, construct error analysis (`results/transformer_v2/error_analysis.csv`), and run ablation study (`results/transformer_v2/ablation_results.csv`).
4. **Conditional Inference Update (Phase 8–9)**: Update `src/inference.py` and `frontend/app.py` only if V2 empirically outperforms V1.
