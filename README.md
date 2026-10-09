# Legal RAG — 47,400 Indian Supreme Court Judgments Pipeline

A modular, multi-phase Legal Retrieval-Augmented Generation (RAG) system built for high-accuracy legal research over Indian Supreme Court judgment PDFs.

---

## 🏗️ Project Architecture & Layout

```text
legal_ai/
├── config.yaml               # Global configuration (paths, models, chunk size, device)
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation & design choices
├── ingest/                   # Phase 1: Ingestion & Document Processing
│   ├── __init__.py
│   ├── pdf_detector.py       # Native vs Scanned PDF inspection
│   ├── extractor.py          # PDF text extraction & OCR quality evaluation
│   ├── cleaner.py            # Header/footer/page-number stripping preserving page map
│   ├── structure_parser.py   # Legal metadata & section tagging
│   ├── chunker.py            # Paragraph chunking (~200-400 tokens) with page range
│   └── pipeline.py           # Ingestion orchestrator
├── index/                    # Phase 2: Vector DB (Qdrant/Chroma) & BM25 indexing
├── retrieval/                # Phase 3: Dense + Lexical Hybrid Search & RRF
├── rerank/                   # Phase 3: Cross-Encoder reranking
├── verify/                   # Phase 4: Deterministic quote check & claim verification
├── generate/                 # Phase 4: Grounded LLM answer generation
├── eval/                     # Phase 5: Evaluation framework & gold queries runner
├── app/                      # CLI & API interface
│   ├── __init__.py
│   └── cli.py                # Command Line Interface (python -m app.cli ...)
├── tests/                    # Automated Unit & Integration Tests
│   └── test_chunker.py       # Chunker page-preservation tests
├── data/
│   ├── sample_pdfs/          # Sample PDF dataset for testing (50 PDFs)
│   └── processed/            # Output JSON chunks and structured data
└── reports/                  # Generated inspection reports (OCR quality report)
```

---

## ⚙️ Device Placement & Hardware Configuration

Configured strictly via `config.yaml`:
- **Hardware Profile:** Windows Laptop, NVIDIA RTX 3050 (4 GB VRAM).
- **VRAM Safety:** Reranker and Embedder run on CPU by default to prevent VRAM out-of-memory errors when local LLM (Ollama `qwen2.5:3b`) is running.
- **Configurability:** All paths, model names, chunk size parameters, and top-k values live in `config.yaml`.

---

## 🚀 Quick Start (Phase 1)

### 1. Run Phase 1 Ingestion Pipeline CLI
```bash
python -m app.cli ingest --limit 50
```

### 2. Run Automated Tests
```bash
python -m unittest tests/test_chunker.py
```

---

## 📊 Phase 1 Deliverables Summary
1. **PDF Detection:** Inspects text density per page to classify Native vs Scanned PDFs.
2. **Page Mapping:** Strips running headers ("SUPREME COURT OF INDIA") and running footers ("Page X of Y") without destroying original page index tracking (`page_start`, `page_end`).
3. **Legal Metadata Parsing:** Extracts `case_name`, `date`, `bench`, and `citation`, fallback to `None`/`Unknown`.
4. **Paragraph Chunking:** Produces ~200-400 token chunks retaining exact metadata and section type.
5. **OCR Quality Reporting:** Saves `reports/ocr_quality_report.md` flagging low-quality scanned documents.
