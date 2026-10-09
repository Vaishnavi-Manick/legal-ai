import os
import logging
from typing import Dict, Any, List
import pypdf

logger = logging.getLogger("LegalRAG.Ingest.Detector")


class PDFDetector:
    """Detects whether a PDF is native text or scanned image-based."""

    def __init__(self, min_text_chars_per_page: int = 50):
        self.min_text_chars_per_page = min_text_chars_per_page

    def inspect_pdf(self, pdf_path: str) -> Dict[str, Any]:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        page_char_counts: List[int] = []
        is_corrupt = False

        try:
            reader = pypdf.PdfReader(pdf_path)
            num_pages = len(reader.pages)

            for idx, page in enumerate(reader.pages):
                try:
                    text = page.extract_text() or ""
                    clean_chars = len(text.strip())
                    page_char_counts.append(clean_chars)
                except Exception as e:
                    logger.warning(f"Error reading page {idx+1} in {pdf_path}: {e}")
                    page_char_counts.append(0)

        except Exception as e:
            logger.error(f"Corrupt or unreadable PDF '{pdf_path}': {e}")
            num_pages = 0
            is_corrupt = True

        avg_chars = (sum(page_char_counts) / num_pages) if num_pages > 0 else 0
        is_scanned = (avg_chars < self.min_text_chars_per_page) and not is_corrupt

        return {
            "source_pdf_path": os.path.abspath(pdf_path),
            "filename": os.path.basename(pdf_path),
            "num_pages": num_pages,
            "page_char_counts": page_char_counts,
            "avg_chars_per_page": round(avg_chars, 2),
            "is_scanned": is_scanned,
            "is_corrupt": is_corrupt
        }

