import os
import sys
import glob
import json
import yaml
import logging
from typing import Dict, Any, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ingest.pdf_detector import PDFDetector
from ingest.extractor import PDFExtractor
from ingest.cleaner import TextCleaner
from ingest.structure_parser import LegalStructureParser
from ingest.chunker import ParagraphChunker

logger = logging.getLogger("LegalRAG.Ingest.Pipeline")


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    full_path = os.path.abspath(config_path)
    if not os.path.exists(full_path):
        full_path = os.path.join(PROJECT_ROOT, "config.yaml")
    with open(full_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class IngestionPipeline:
    """Orchestrates Steps 1 to 5 of Phase 1 Ingestion Pipeline."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        ing_cfg = config.get("ingestion", {})
        chk_cfg = ing_cfg.get("chunking", {})

        self.detector = PDFDetector(min_text_chars_per_page=ing_cfg.get("min_text_chars_per_page", 50))
        self.extractor = PDFExtractor(scanned_ocr_threshold=ing_cfg.get("scanned_ocr_threshold", 0.4))
        self.cleaner = TextCleaner()
        self.parser = LegalStructureParser()
        self.chunker = ParagraphChunker(
            min_chunk_tokens=chk_cfg.get("min_chunk_tokens", 150),
            max_chunk_tokens=chk_cfg.get("max_chunk_tokens", 350),
            overlap_tokens=chk_cfg.get("overlap_tokens", 40)
        )

    def run_on_directory(self, pdf_dir: str, limit: int = 50) -> Dict[str, Any]:
        pdf_paths = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
        if limit and limit > 0:
            pdf_paths = pdf_paths[:limit]

        logger.info(f"Processing {len(pdf_paths)} PDFs from '{pdf_dir}'...")

        all_documents: List[Dict[str, Any]] = []
        all_chunks: List[Dict[str, Any]] = []
        ocr_report_entries: List[Dict[str, Any]] = []

        for idx, pdf_path in enumerate(pdf_paths):
            try:
                # Step 1: PDF Detection
                det_res = self.detector.inspect_pdf(pdf_path)

                # Step 2: Extraction & OCR Quality Check
                ext_res = self.extractor.extract_document(det_res)

                # Step 3: Cleaning (preserve page mapping)
                clean_res = self.cleaner.clean_document(ext_res)

                # Step 4: Legal Structure & Metadata Parsing
                parsed_res = self.parser.parse_document(clean_res)

                # Step 5: Paragraph Chunking
                doc_chunks = self.chunker.create_chunks(parsed_res)

                all_documents.append(parsed_res)
                all_chunks.extend(doc_chunks)

                ocr_report_entries.append({
                    "filename": ext_res["filename"],
                    "source_pdf_path": ext_res["source_pdf_path"],
                    "is_scanned": ext_res["is_scanned"],
                    "is_corrupt": ext_res["is_corrupt"],
                    "ocr_quality_score": ext_res["ocr_quality_score"],
                    "is_low_quality": ext_res["is_low_quality"],
                    "num_pages": det_res["num_pages"],
                    "chunks_count": len(doc_chunks)
                })

                logger.info(f"[{idx+1}/{len(pdf_paths)}] Processed {ext_res['filename']} -> {len(doc_chunks)} chunks (OCR Score: {ext_res['ocr_quality_score']})")

            except Exception as e:
                logger.error(f"Failed to process '{pdf_path}': {e}", exc_info=True)
                ocr_report_entries.append({
                    "filename": os.path.basename(pdf_path),
                    "source_pdf_path": pdf_path,
                    "is_scanned": False,
                    "is_corrupt": True,
                    "ocr_quality_score": 0.0,
                    "is_low_quality": True,
                    "num_pages": 0,
                    "chunks_count": 0
                })

        # Save output artifacts
        proc_dir = os.path.abspath(self.config["paths"]["processed_dir"])
        rep_dir = os.path.abspath(self.config["paths"]["reports_dir"])
        os.makedirs(proc_dir, exist_ok=True)
        os.makedirs(rep_dir, exist_ok=True)

        chunks_out_path = os.path.join(proc_dir, "chunks.json")
        with open(chunks_out_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2)

        # Build OCR Quality Report
        total_docs = len(ocr_report_entries)
        scanned_count = sum(1 for e in ocr_report_entries if e["is_scanned"])
        native_count = total_docs - scanned_count
        low_quality_count = sum(1 for e in ocr_report_entries if e["is_low_quality"])

        report_summary = {
            "total_documents_processed": total_docs,
            "native_pdfs_count": native_count,
            "scanned_pdfs_count": scanned_count,
            "low_quality_flagged_count": low_quality_count,
            "total_chunks_generated": len(all_chunks),
            "documents": ocr_report_entries
        }

        report_json_path = os.path.join(rep_dir, "ocr_quality_report.json")
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(report_summary, f, indent=2)

        report_md_path = os.path.join(rep_dir, "ocr_quality_report.md")
        self._write_report_markdown(report_summary, report_md_path)

        logger.info(f"Ingestion pipeline complete. Generated {len(all_chunks)} chunks across {total_docs} documents.")
        return {
            "documents": all_documents,
            "chunks": all_chunks,
            "report": report_summary
        }

    def _write_report_markdown(self, summary: Dict[str, Any], filepath: str):
        md = f"""# Legal RAG Ingestion Pipeline — OCR Quality Report

## Overview & Summary Statistics
- **Total Documents Processed:** {summary['total_documents_processed']}
- **Native PDFs Identified:** {summary['native_pdfs_count']}
- **Scanned PDFs Identified:** {summary['scanned_pdfs_count']}
- **Low Quality / OCR Flagged Documents:** {summary['low_quality_flagged_count']}
- **Total Chunks Generated:** {summary['total_chunks_generated']}

## Document Inspection Breakdown

| Filename | Type | Pages | Chunks | OCR Quality Score | Status |
|---|---|---|---|---|---|
"""
        for doc in summary["documents"]:
            doc_type = "Scanned" if doc["is_scanned"] else ("Corrupt" if doc["is_corrupt"] else "Native")
            status = "⚠️ LOW QUALITY" if doc["is_low_quality"] else "✅ OK"
            md += f"| `{doc['filename']}` | {doc_type} | {doc['num_pages']} | {doc['chunks_count']} | {doc['ocr_quality_score']:.4f} | {status} |\n"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(md)

