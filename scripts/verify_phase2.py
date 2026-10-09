import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ingest.pipeline import load_config
from index.dense_indexer import DenseIndexer
from index.bm25_indexer import BM25Indexer

config = load_config("config.yaml")
dense_idx = DenseIndexer(config)
bm25_idx = BM25Indexer(config)

queries = [
    "right to privacy under Article 21",
    "preventive detention and natural justice"
]

for q in queries:
    print("=" * 70)
    print(f"QUERY: '{q}'")
    print("\n--- DENSE SEARCH TOP-3 ---")
    d_res = dense_idx.search(q, top_k=3)
    for rank, (chk, score) in enumerate(d_res, 1):
        print(f"  #{rank} | Score: {score:.4f} | Chunk ID: {chk['chunk_id']} | Case: {chk['case_name']}")

    print("\n--- BM25 LEXICAL SEARCH TOP-3 ---")
    b_res = bm25_idx.search(q, top_k=3)
    for rank, (chk, score) in enumerate(b_res, 1):
        print(f"  #{rank} | Score: {score:.4f} | Chunk ID: {chk['chunk_id']} | Case: {chk['case_name']}")

