import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from retrieval.router import CitationRouter
from retrieval.query_rewriter import LLMQueryRewriter
from retrieval.fusion import ReciprocalRankFusion


class TestQueryPipelineComponents(unittest.TestCase):

    def test_citation_router(self):
        router = CitationRouter()

        res1 = router.route_query("2019 SCC 123")
        self.assertTrue(res1["is_citation"])
        self.assertTrue(res1["skip_rewriting"])

        res2 = router.route_query("Writ Petition No. 117 of 1973")
        self.assertTrue(res1["is_citation"])

        res3 = router.route_query("right to privacy under Article 21")
        self.assertFalse(res3["skip_rewriting"])

    def test_query_rewriter_fallback(self):
        config = {
            "environment": {"ollama_base_url": "http://localhost:11434", "llm_model": "qwen2.5:3b"},
            "query_pipeline": {"query_rewrite_enabled": False}
        }
        rewriter = LLMQueryRewriter(config)
        res = rewriter.rewrite_query("principles of natural justice")

        self.assertFalse(res["was_rewritten"])
        self.assertEqual(res["original_query"], res["rewritten_query"])

    def test_rrf_fusion(self):
        fusion = ReciprocalRankFusion(rrf_k=60)
        bm25_cands = [
            ({"chunk_id": "c1", "text": "chunk 1 text"}, 10.5),
            ({"chunk_id": "c2", "text": "chunk 2 text"}, 8.2)
        ]
        dense_cands = [
            ({"chunk_id": "c2", "text": "chunk 2 text"}, 0.95),
            ({"chunk_id": "c3", "text": "chunk 3 text"}, 0.88)
        ]

        fused = fusion.fuse_rankings(bm25_cands, dense_cands, top_k=3)
        self.assertTrue(len(fused) > 0)
        # c2 appears in both lists, so its RRF score should be highest
        self.assertEqual(fused[0][0]["chunk_id"], "c2")


if __name__ == "__main__":
    unittest.main()

