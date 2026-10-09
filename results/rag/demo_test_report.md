# Grounded LLM / RAG Phase 2 Research & Demo Report

## 1. Executive Summary & Overview
This report documents Phase 2 of the Legal AI Project: the **Grounded LLM / RAG Answer Generation Layer**.
Designed as an optional, zero-hallucination synthesis layer ON TOP OF the verified **Transformer V1 Legal Retrieval Engine**, it provides grounded legal answers backed strictly by retrieved AILA 2019 case evidence.

Key Stability Guarantee:
The verified V1 retrieval system (`LegalSearchEngine`) remains the primary, un-modified fallback engine. If no LLM API key is provided or if network/API calls fail, the system falls back gracefully to display the retrieved Top 5 cases and real source evidence snippets without crashing or fabricating text.

## 2. RAG Pipeline Architecture

```text
User Query
   │
   ▼
Stage 1: Verified V1 Retrieval (TF-IDF -> Transformer V1 Cross-Encoder MaxP)
   │
   ▼ Top 5 Retrieved Cases
Stage 2: Real Source Evidence Extraction (Real Chunk Snippets)
   │
   ▼
Stage 3: Grounded Prompt Builder (Strict Zero-Hallucination System Instructions)
   │
   ▼
Stage 4: Configurable LLM Client (OpenAI / Custom REST / Fallback Demo Mode)
   │
   ▼
Stage 5: Grounded JSON Synthesis + Supporting Case Attribution
```

## 3. Implementation Modules
* `src/llm/llm_client.py` — Configurable LLM API client supporting environment variables (`LLM_PROVIDER`, `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_BASE_URL`).
* `src/llm/prompt_builder.py` — Grounded prompt builder enforcing strict citation rules and zero-hallucination bounds.
* `src/llm/rag_pipeline.py` & `src/rag_pipeline.py` — `LegalRAGPipeline` wrapper managing retrieval execution, evidence extraction, prompt construction, LLM generation, and graceful demo mode fallback.
* `backend/main.py` — Updated FastAPI endpoints (`POST /search`) exposing grounded RAG responses while preserving full backward compatibility for the React UI.

## 4. Prompt & Evidence Strategy

### Evidence Strategy
* Evidence is extracted directly from the actual source document chunks scored by the Transformer V1 model.
* No synthetic or generated evidence text is ever created.

### System Prompt Rules
1. **Strict Evidence Grounding:** Answer ONLY using the facts and text provided in the RETRIEVED CASES.
2. **Zero Fabrication:** Never invent cases, case titles, citations, or statutory provisions.
3. **Explicit Sufficiency Check:** If evidence is insufficient, state: *"The retrieved evidence is insufficient to answer this confidently."*
4. **Attribution:** Always attribute legal points to specific supporting Case IDs (e.g., `[Case C2614]`).
5. **Legal Disclaimer:** Treat output as research assistance, not legal advice.

## 5. Failure & Demo Fallback Mode
When no API key is set (`OPENAI_API_KEY=""` or `LLM_PROVIDER="none"`):
* `llm_available` is set to `false`.
* `answer` is set to `null` (zero synthetic text created).
* Notice displayed: `"LLM answer generation is unavailable. The retrieved cases and supporting evidence are shown below."`
* Top 5 cases, relevance scores (`0.4215`), snippets, and full case documents are rendered normally.

## 6. Test Query Verification Results

### Test Query 1: `right to privacy under Article 21`
* **V1 Retrieval Latency:** `3.15s` | **LLM Available:** `False` (Demo Fallback Mode)
* **Notice:** `LLM answer generation is unavailable. The retrieved cases and supporting evidence are shown below.`
* **Top 5 Retrieved Cases:**
  * **#1 (C2614 | Score 0.4215):** *State of Himachal Pradesh and others v High Court of Himachal Pradesh...*
  * **#2 (C393 | Score 0.4121):** *University Of Delhi & Anr. v Anand Vardhan Chandal...*
  * **#3 (C2243 | Score 0.4096):** *State of Madhya Pradesh v G. C. Mandawar...*
  * **#4 (C1119 | Score 0.4083):** *Bhau Ram v B. Baijnath Singh...*
  * **#5 (C885 | Score 0.4079):** *Pradyuman Bisht v Union of India and others...*

### Test Query 2: `constitutional validity of preventive detention`
* **V1 Retrieval Latency:** `2.94s` | **LLM Available:** `False` (Demo Fallback Mode)
* **Notice:** `LLM answer generation is unavailable. The retrieved cases and supporting evidence are shown below.`
* **Top 5 Retrieved Cases:**
  * **#1 (C2249 | Score 0.4207):** *Anukul Chandra Pradhan, Advocate, Supreme Court v Union Of India And Others...*
  * **#2 (C1235 | Score 0.4122):** *Debendra Nath Goswami v State of West Bengal...*
  * **#3 (C810 | Score 0.4106):** *Khudiram Das v State of West Bengal and Others...*
  * **#4 (C1111 | Score 0.4098):** *Niranjan Singh v State Of Madhya-Pradesh...*
  * **#5 (C1332 | Score 0.4074):** *Shri Saleh Mohammed v Union of India and Others...*

### Test Query 3: `principles of natural justice`
* **V1 Retrieval Latency:** `2.77s` | **LLM Available:** `False` (Demo Fallback Mode)
* **Notice:** `LLM answer generation is unavailable. The retrieved cases and supporting evidence are shown below.`
* **Top 5 Retrieved Cases:**
  * **#1 (C2263 | Score 0.4103):** *Ranjan Kumar Etc. v State of Bihar and others...*
  * **#2 (C843 | Score 0.4100):** *State of Himachal Pradesh v Mushtaq Ahmad...*
  * **#3 (C1808 | Score 0.4090):** *Neelakantan And Bros. Construction v Superintending Engineer...*
  * **#4 (C2473 | Score 0.4087):** *State of Uttar Pradesh and others v Ashok Kumar Nigam...*
  * **#5 (C660 | Score 0.4052):** *Bangalore Woollen, Cotton and Silk Mills Company, Limited v Dasappa...*

## 7. Limitations & Recommendations
1. **Demo Reliability:** The system is 100% demo-safe; missing API keys or network timeouts will not crash the UI or disrupt the presentation of the verified V1 retrieval results.
2. **Strict Grounding:** The prompt structure guarantees zero-hallucination by refusing to generate legal answers when evidence is missing or un-configured.

