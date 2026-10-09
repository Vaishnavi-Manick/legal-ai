import os
import re
import logging
from typing import Dict, Any, List
import pdfplumber
import pypdf
from PIL import Image

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

logger = logging.getLogger("LegalRAG.Ingest.Extractor")


class PDFExtractor:
    """Extracts raw text page by page from native or scanned PDFs and evaluates OCR quality."""

    def __init__(self, scanned_ocr_threshold: float = 0.4):
        self.scanned_ocr_threshold = scanned_ocr_threshold

    def _calculate_ocr_quality(self, text: str) -> float:
        """Computes a quality score (0.0 to 1.0) based on character distributions and printable word ratios."""
        if not text or not text.strip():
            return 0.0

        total_chars = len(text)
        alpha_num_chars = len(re.findall(r'[a-zA-Z0-9\s.,;:\'"()-]', text))
        char_score = alpha_num_chars / max(total_chars, 1)

        words = text.split()
        if not words:
            return 0.0

        valid_words = [w for w in words if re.match(r'^[a-zA-Z0-9.,;:\'"()-]+$', w)]
        word_score = len(valid_words) / max(len(words), 1)

        return round(0.5 * char_score + 0.5 * word_score, 4)

    def extract_document(self, detection_result: Dict[str, Any]) -> Dict[str, Any]:
        pdf_path = detection_result["source_pdf_path"]
        is_scanned = detection_result.get("is_scanned", False)
        is_corrupt = detection_result.get("is_corrupt", False)

        pages_data: List[Dict[str, Any]] = []
        doc_ocr_score = 1.0  # Native PDFs default to 1.0 quality

        if is_corrupt:
            logger.error(f"Skipping corrupt PDF: {pdf_path}")
            return {
                "source_pdf_path": pdf_path,
                "filename": detection_result["filename"],
                "pages": [],
                "ocr_quality_score": 0.0,
                "is_low_quality": True,
                "is_scanned": is_scanned,
                "is_corrupt": True
            }

        if not is_scanned:
            # Native PDF extraction via pdfplumber
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for idx, page in enumerate(pdf.pages):
                        page_num = idx + 1
                        raw_text = page.extract_text() or ""
                        pages_data.append({
                            "page_num": page_num,
                            "raw_text": raw_text
                        })
            except Exception as e:
                logger.warning(f"pdfplumber failed for '{pdf_path}', falling back to pypdf: {e}")
                pages_data = []
                try:
                    reader = pypdf.PdfReader(pdf_path)
                    for idx, page in enumerate(reader.pages):
                        pages_data.append({
                            "page_num": idx + 1,
                            "raw_text": page.extract_text() or ""
                        })
                except Exception as ex:
                    logger.error(f"pypdf fallback also failed for '{pdf_path}': {ex}")

        else:
            # Scanned PDF OCR extraction
            logger.info(f"Extracting scanned PDF using OCR: {pdf_path}")
            page_ocr_scores = []
            
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for idx, page in enumerate(pdf.pages):
                        page_num = idx + 1
                        ocr_text = ""
                        try:
                            pil_img = page.to_image(resolution=150).original
                            if PYTESSERACT_AVAILABLE:
                                try:
                                    ocr_text = pytesseract.image_to_string(pil_img)
                                except Exception as terr:
                                    logger.warning(f"tesseract execution error: {terr}")
                                    ocr_text = page.extract_text() or ""
                            else:
                                ocr_text = page.extract_text() or ""
                        except Exception as img_err:
                            logger.warning(f"Failed image rendering on page {page_num}: {img_err}")
                            ocr_text = page.extract_text() or ""

                        pages_data.append({
                            "page_num": page_num,
                            "raw_text": ocr_text
                        })
                        page_ocr_scores.append(self._calculate_ocr_quality(ocr_text))

            except Exception as e:
                logger.error(f"OCR extraction failed for '{pdf_path}': {e}")

            doc_ocr_score = (sum(page_ocr_scores) / len(page_ocr_scores)) if page_ocr_scores else 0.0

        is_low_quality = doc_ocr_score < self.scanned_ocr_threshold

        return {
            "source_pdf_path": pdf_path,
            "filename": detection_result["filename"],
            "pages": pages_data,
            "ocr_quality_score": round(doc_ocr_score, 4),
            "is_low_quality": is_low_quality,
            "is_scanned": is_scanned,
            "is_corrupt": False
        }

