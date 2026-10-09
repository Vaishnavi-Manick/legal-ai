import os
import hashlib
import logging
from typing import Dict, Any, List

logger = logging.getLogger("LegalRAG.Ingest.Chunker")


class ParagraphChunker:
    """Paragraph-level chunker (~200 to 400 tokens, overlap) preserving exact page_start and page_end."""

    def __init__(self, min_chunk_tokens: int = 150, max_chunk_tokens: int = 350, overlap_tokens: int = 40):
        self.min_chunk_tokens = min_chunk_tokens
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap_tokens = overlap_tokens

    def _estimate_tokens(self, text: str) -> int:
        # Approximate 1 token = ~0.75 words, or word count * 1.3
        words = text.split()
        return int(len(words) * 1.3)

    def create_chunks(self, parsed_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        case_id = parsed_doc["case_id"]
        case_name = parsed_doc.get("case_name", "Unknown Case")
        date = parsed_doc.get("date")
        bench = parsed_doc.get("bench")
        citation = parsed_doc.get("citation")
        source_pdf_path = parsed_doc["source_pdf_path"]

        parsed_pages = parsed_doc.get("parsed_pages", [])
        if not parsed_pages:
            return []

        # Flatten paragraphs with page numbers
        flattened_items: List[Dict[str, Any]] = []
        for p in parsed_pages:
            page_num = p["page_num"]
            for para in p.get("paragraphs", []):
                text = para["text"].strip()
                if text:
                    flattened_items.append({
                        "page_num": page_num,
                        "text": text,
                        "section_type": para["section_type"]
                    })

        chunks: List[Dict[str, Any]] = []
        curr_words: List[str] = []
        curr_pages: List[int] = []
        curr_section_types: List[str] = []

        chunk_idx = 0

        for item in flattened_items:
            words = item["text"].split()
            page_num = item["page_num"]
            sec_type = item["section_type"]

            curr_words.extend(words)
            curr_pages.extend([page_num] * len(words))
            curr_section_types.append(sec_type)

            curr_token_count = int(len(curr_words) * 1.3)

            if curr_token_count >= self.max_chunk_tokens:
                chunk_text = " ".join(curr_words)
                p_start = curr_pages[0] if curr_pages else 1
                p_end = curr_pages[-1] if curr_pages else p_start
                main_sec = max(set(curr_section_types), key=curr_section_types.count) if curr_section_types else "unknown"

                chunk_id = f"{case_id}_chk_{chunk_idx:04d}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "case_id": case_id,
                    "case_name": case_name,
                    "date": date,
                    "bench": bench,
                    "citation": citation,
                    "section_type": main_sec,
                    "page_start": p_start,
                    "page_end": p_end,
                    "source_pdf_path": source_pdf_path,
                    "text": chunk_text,
                    "token_count": curr_token_count
                })

                chunk_idx += 1

                # Overlap handling
                overlap_word_count = int(self.overlap_tokens / 1.3)
                if len(curr_words) > overlap_word_count:
                    curr_words = curr_words[-overlap_word_count:]
                    curr_pages = curr_pages[-overlap_word_count:]
                    curr_section_types = curr_section_types[-1:]
                else:
                    curr_words = []
                    curr_pages = []
                    curr_section_types = []

        # Remaining words flush
        if curr_words:
            chunk_text = " ".join(curr_words)
            p_start = curr_pages[0] if curr_pages else 1
            p_end = curr_pages[-1] if curr_pages else p_start
            main_sec = max(set(curr_section_types), key=curr_section_types.count) if curr_section_types else "unknown"

            chunk_id = f"{case_id}_chk_{chunk_idx:04d}"
            chunks.append({
                "chunk_id": chunk_id,
                "case_id": case_id,
                "case_name": case_name,
                "date": date,
                "bench": bench,
                "citation": citation,
                "section_type": main_sec,
                "page_start": p_start,
                "page_end": p_end,
                "source_pdf_path": source_pdf_path,
                "text": chunk_text,
                "token_count": int(len(curr_words) * 1.3)
            })

        return chunks

