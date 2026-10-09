import os
import re
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("LegalRAG.Ingest.StructureParser")


class LegalStructureParser:
    """Extracts legal case metadata and tags section types (facts, issues, arguments, reasoning, order, dissent)."""

    CASE_NAME_PATTERN = re.compile(r"([A-Z][A-Za-z0-9\s.,&'()-]+?\s+(?:v\.|vs\.|Versus|V\.)\s+[A-Z][A-Za-z0-9\s.,&'()-]+)", re.IGNORECASE)
    DATE_PATTERN = re.compile(r"(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{4})", re.IGNORECASE)
    BENCH_PATTERN = re.compile(r"(?:Delivered by|CORAM|Bench|BEFORE)\s*[:;-]?\s*([A-Z][A-Za-z0-9\s.,;()-]+)", re.IGNORECASE)
    CITATION_PATTERN = re.compile(r"(Writ Petition[^\n]+|Civil Appeal[^\n]+|Criminal Appeal[^\n]+|\d{4}\s+SCC[^\n]+|\d{4}\s+AIR[^\n]+)", re.IGNORECASE)

    SECTION_KEYWORDS = {
        "facts": [r"fact", r"background", r"petitioner prays", r"filed", r"enrolled", r"prosecuted"],
        "issues": [r"issue", r"question of law", r"point for determination", r"whether"],
        "arguments": [r"argued", r"contended", r"contention", r"submitted", r"learned counsel"],
        "reasoning": [r"considered", r"holding", r"held", r"principle", r"section", r"article", r"precedent"],
        "order": [r"order accordingly", r"dismissed", r"allowed", r"directed", r"rejected", r"release"],
        "dissent": [r"dissent", r"disagree", r"minority view"]
    }

    def _extract_metadata(self, full_text: str) -> Dict[str, Optional[str]]:
        lines = [line.strip() for line in full_text.split("\n") if line.strip()]
        first_few = "\n".join(lines[:10])

        # 1. Case Name
        case_name_match = self.CASE_NAME_PATTERN.search(first_few)
        case_name = case_name_match.group(1).strip() if case_name_match else (lines[0] if lines else "Unknown Case")

        # 2. Date
        date_match = self.DATE_PATTERN.search(first_few)
        date_str = date_match.group(1).strip() if date_match else None

        # 3. Bench
        bench_match = self.BENCH_PATTERN.search(first_few)
        bench_str = bench_match.group(1).strip() if bench_match else None

        # 4. Citation
        citation_match = self.CITATION_PATTERN.search(first_few)
        citation_str = citation_match.group(1).strip() if citation_match else None

        return {
            "case_name": case_name,
            "date": date_str,
            "bench": bench_str,
            "citation": citation_str
        }

    def detect_section_type(self, text: str) -> str:
        text_lower = text.lower()
        for sec_type, keywords in self.SECTION_KEYWORDS.items():
            for kw in keywords:
                if re.search(r"\b" + kw + r"\b", text_lower):
                    return sec_type
        return "unknown"

    def parse_document(self, cleaned_doc: Dict[str, Any]) -> Dict[str, Any]:
        pages = cleaned_doc.get("pages", [])
        full_text = "\n".join([p["cleaned_text"] for p in pages if p["cleaned_text"]])

        metadata = self._extract_metadata(full_text)

        # Derive case_id from filename or case_name
        filename = cleaned_doc["filename"]
        case_id = os.path.splitext(filename)[0]

        parsed_pages = []
        for p in pages:
            lines = p.get("lines", [])
            paragraphs = []
            curr_para = []

            for line in lines:
                curr_para.append(line)
                if line.endswith((".", ":", ";")) or len(curr_para) >= 4:
                    para_text = " ".join(curr_para).strip()
                    if para_text:
                        sec_type = self.detect_section_type(para_text)
                        paragraphs.append({
                            "text": para_text,
                            "section_type": sec_type
                        })
                    curr_para = []

            if curr_para:
                para_text = " ".join(curr_para).strip()
                if para_text:
                    sec_type = self.detect_section_type(para_text)
                    paragraphs.append({
                        "text": para_text,
                        "section_type": sec_type
                    })

            parsed_pages.append({
                "page_num": p["page_num"],
                "paragraphs": paragraphs
            })

        return {
            "case_id": case_id,
            "case_name": metadata["case_name"],
            "date": metadata["date"],
            "bench": metadata["bench"],
            "citation": metadata["citation"],
            "source_pdf_path": cleaned_doc["source_pdf_path"],
            "filename": cleaned_doc["filename"],
            "ocr_quality_score": cleaned_doc["ocr_quality_score"],
            "is_low_quality": cleaned_doc["is_low_quality"],
            "is_scanned": cleaned_doc["is_scanned"],
            "is_corrupt": cleaned_doc["is_corrupt"],
            "parsed_pages": parsed_pages
        }
