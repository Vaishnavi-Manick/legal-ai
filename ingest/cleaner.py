import re
import logging
from typing import Dict, Any, List

logger = logging.getLogger("LegalRAG.Ingest.Cleaner")


class TextCleaner:
    """Strips running headers, footers, and page numbers while preserving page mapping."""

    HEADER_PATTERNS = [
        re.compile(r"^SUPREME COURT OF INDIA.*$", re.IGNORECASE),
        re.compile(r"^RECORD OF PROCEEDINGS.*$", re.IGNORECASE),
        re.compile(r"^IN THE SUPREME COURT OF INDIA.*$", re.IGNORECASE),
        re.compile(r"^\[SCANNED JUDGMENT DOCUMENT.*\]$", re.IGNORECASE),
    ]

    FOOTER_PATTERNS = [
        re.compile(r"^Page \d+ of \d+$", re.IGNORECASE),
        re.compile(r"^Page \d+$", re.IGNORECASE),
        re.compile(r"^\d+ of \d+$", re.IGNORECASE),
        re.compile(r"^CONFIDENTIAL.*$", re.IGNORECASE),
        re.compile(r"^\d+\s*$", re.IGNORECASE),  # Standalone page numbers
    ]

    def clean_document(self, extracted_doc: Dict[str, Any]) -> Dict[str, Any]:
        cleaned_pages: List[Dict[str, Any]] = []

        for p in extracted_doc.get("pages", []):
            page_num = p["page_num"]
            raw_text = p.get("raw_text", "")
            lines = raw_text.split("\n")
            cleaned_lines = []

            for idx, line in enumerate(lines):
                trimmed = line.strip()
                if not trimmed:
                    continue

                # Strip first 2 lines if header pattern matches
                if idx < 3 and any(pattern.match(trimmed) for pattern in self.HEADER_PATTERNS):
                    continue

                # Strip last 3 lines if footer pattern matches
                if idx >= max(0, len(lines) - 3) and any(pattern.match(trimmed) for pattern in self.FOOTER_PATTERNS):
                    continue

                cleaned_lines.append(trimmed)

            cleaned_text = "\n".join(cleaned_lines)
            cleaned_pages.append({
                "page_num": page_num,
                "cleaned_text": cleaned_text,
                "lines": cleaned_lines
            })

        return {
            "source_pdf_path": extracted_doc["source_pdf_path"],
            "filename": extracted_doc["filename"],
            "ocr_quality_score": extracted_doc.get("ocr_quality_score", 1.0),
            "is_low_quality": extracted_doc.get("is_low_quality", False),
            "is_scanned": extracted_doc.get("is_scanned", False),
            "is_corrupt": extracted_doc.get("is_corrupt", False),
            "pages": cleaned_pages
        }

