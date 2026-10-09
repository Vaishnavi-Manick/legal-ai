import os
import sys
import unittest
import shutil

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from index.dense_indexer import DenseIndexer
from index.bm25_indexer import BM25Indexer


class TestIndexingPipeline(unittest.TestCase):

    def setUp(self):
        self.test_dir = os.path.join(PROJECT_ROOT, "tests", "scratch_test_index")
        os.makedirs(self.test_dir, exist_ok=True)

        self.config = {
            "environment": {"device": "cpu", "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"},
            "paths": {
                "vector_db_dir": os.path.join(self.test_dir, "dense"),
                "bm25_index_dir": os.path.join(self.test_dir, "bm25")
            }
        }

        self.sample_chunks = [
            {
                "chunk_id": "test_c1_chk_0000",
                "case_id": "test_c1",
                "case_name": "State v. Defendant",
                "text": "Right to privacy under Article 21 of the Indian Constitution is a fundamental right."
            },
            {
                "chunk_id": "test_c2_chk_0000",
                "case_id": "test_c2",
                "case_name": "Public v. Union",
                "text": "Preventive detention must comply with procedural natural justice guidelines."
            },
            {
                "chunk_id": "test_c3_chk_0000",
                "case_id": "test_c3",
                "case_name": "Revenue v. Assessee",
                "text": "Income tax assessment and statutory appeal process under the Income Tax Act."
            }
        ]

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_dense_and_bm25_indexing(self):
        dense_idx = DenseIndexer(self.config)
        bm25_idx = BM25Indexer(self.config)

        dense_meta = dense_idx.build_index_with_checkpoint(self.sample_chunks, batch_size=1)
        bm25_meta = bm25_idx.build_index(self.sample_chunks)

        # 1-to-1 chunk ID alignment
        self.assertEqual(dense_meta["chunk_ids"], bm25_meta["chunk_ids"])
        self.assertEqual(len(dense_meta["chunk_ids"]), 3)

        # Test Dense Search
        d_results = dense_idx.search("privacy article 21", top_k=1)
        self.assertEqual(len(d_results), 1)
        self.assertEqual(d_results[0][0]["chunk_id"], "test_c1_chk_0000")

        # Test BM25 Search
        b_results = bm25_idx.search("preventive detention", top_k=1)
        self.assertEqual(len(b_results), 1)
        self.assertEqual(b_results[0][0]["chunk_id"], "test_c2_chk_0000")


if __name__ == "__main__":
    unittest.main()

