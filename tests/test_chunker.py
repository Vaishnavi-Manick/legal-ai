import unittest
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ingest.chunker import ParagraphChunker


class TestParagraphChunker(unittest.TestCase):

    def test_chunker_preserves_page_numbers_and_metadata(self):
        parsed_doc = {
            "case_id": "test_case_001",
            "case_name": "State of Maharashtra v. Indian Citizen",
            "date": "15 January 2020",
            "bench": "D.Y. Chandrachud, J.",
            "citation": "2020 SCC 456",
            "source_pdf_path": "/path/to/test.pdf",
            "parsed_pages": [
                {
                    "page_num": 1,
                    "paragraphs": [
                        {"text": "This is paragraph 1 on page 1 detailing the factual background of the case.", "section_type": "facts"},
                        {"text": "Paragraph 2 on page 1 discussing the legal issue.", "section_type": "issues"}
                    ]
                },
                {
                    "page_num": 2,
                    "paragraphs": [
                        {"text": "Paragraph 3 on page 2 detailing judicial arguments and precedents.", "section_type": "arguments"},
                        {"text": "Paragraph 4 on page 2 rendering the final court order and directions.", "section_type": "order"}
                    ]
                }
            ]
        }

        chunker = ParagraphChunker(min_chunk_tokens=10, max_chunk_tokens=50, overlap_tokens=5)
        chunks = chunker.create_chunks(parsed_doc)

        self.assertGreater(len(chunks), 0, "Chunker should produce at least one chunk")

        for chk in chunks:
            self.assertIn("chunk_id", chk)
            self.assertEqual(chk["case_id"], "test_case_001")
            self.assertEqual(chk["case_name"], "State of Maharashtra v. Indian Citizen")
            self.assertEqual(chk["date"], "15 January 2020")
            self.assertEqual(chk["bench"], "D.Y. Chandrachud, J.")
            self.assertEqual(chk["citation"], "2020 SCC 456")
            self.assertIn(chk["page_start"], [1, 2])
            self.assertIn(chk["page_end"], [1, 2])
            self.assertLessEqual(chk["page_start"], chk["page_end"])
            self.assertTrue(len(chk["text"]) > 0)
            self.assertIn(chk["section_type"], ["facts", "issues", "arguments", "reasoning", "order", "dissent", "unknown"])


if __name__ == "__main__":
    unittest.main()

