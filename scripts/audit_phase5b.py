import os
import sys
import json
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ingest.pipeline import load_config
from index.dense_indexer import DenseIndexer
from index.bm25_indexer import BM25Indexer
from retrieval.query_engine import Phase3QueryEngine
from src.inference import LegalSearchEngine

def audit():
    config = load_config("config.yaml")
    hybrid_engine = Phase3QueryEngine(config)
    v1_engine = LegalSearchEngine()

    test_queries = [
        "right to privacy under Article 21",
        "constitutional validity of preventive detention",
        "principles of natural justice"
    ]

    audit_results = []

    print("=" * 85)
    print("                PHASE 5B RETRIEVAL QUALITY AUDIT REPORT")
    print("=" * 85)

    for q in test_queries:
        print(f"\n=================================================================================")
        print(f"AUDIT QUERY: '{q}'")
        print(f"=================================================================================")

        # A. Original V1 Engine Top 5
        v1_results = v1_engine.search(q, top_k=5)

        # B. Hybrid + Transformer Pipeline
        hyb_results = hybrid_engine.search(q)

        # C. BM25 Top 10
        bm25_top_10 = hybrid_engine.bm25_indexer.search(q, top_k=10)

        # D. Dense Top 10
        dense_top_10 = hybrid_engine.dense_indexer.search(q, top_k=10)

        # E. Transformer Scores before Aggregation
        fused_candidates = hyb_results["rrf_candidates"]
        scored_candidates = hybrid_engine.v1_reranker.score_chunks(q, fused_candidates)

        q_audit = {
            "query": q,
            "original_v1_top_5": [
                {"rank": r["rank"], "case_id": r["case_id"], "score": r["relevance_score"], "snippet": r["snippet"][:150]}
                for r in v1_results
            ],
            "bm25_top_10": [
                {"rank": idx+1, "chunk_id": c[0]["chunk_id"], "case_id": c[0]["case_id"], "score": round(c[1], 4)}
                for idx, c in enumerate(bm25_top_10)
            ],
            "dense_top_10": [
                {"rank": idx+1, "chunk_id": c[0]["chunk_id"], "case_id": c[0]["case_id"], "score": round(c[1], 4)}
                for idx, c in enumerate(dense_top_10)
            ],
            "raw_transformer_scores_before_aggregation": [
                {"chunk_id": chunk["chunk_id"], "case_id": chunk["case_id"], "score": round(s, 4)}
                for chunk, s in scored_candidates[:10]
            ],
            "final_maxp_top_5": [
                {"rank": idx+1, "case_id": chunk["case_id"], "best_chunk_id": chunk["chunk_id"], "score": round(s, 4), "snippet": chunk["text"][:150]}
                for idx, (chunk, s) in enumerate(hyb_results["final_top_5"])
            ]
        }

        print("\nA. ORIGINAL V1 ENGINE TOP 5:")
        for r in v1_results:
            print(f"   #{r['rank']} | Case ID: {r['case_id']} | Score: {r['relevance_score']:.4f} | Snippet: \"{r['snippet'][:100]}...\"")

        print("\nB. BM25 RETRIEVAL TOP 5:")
        for idx, (chunk, s) in enumerate(bm25_top_10[:5], 1):
            print(f"   #{idx} | Score: {s:.4f} | Chunk ID: {chunk['chunk_id']} | Case ID: {chunk['case_id']}")

        print("\nC. DENSE RETRIEVAL TOP 5:")
        for idx, (chunk, s) in enumerate(dense_top_10[:5], 1):
            print(f"   #{idx} | Score: {s:.4f} | Chunk ID: {chunk['chunk_id']} | Case ID: {chunk['case_id']}")

        print("\nD. TRANSFORMER SCORES BEFORE AGGREGATION (TOP 5 CHUNKS):")
        for idx, (chunk, s) in enumerate(scored_candidates[:5], 1):
            print(f"   #{idx} | Score: {s:.4f} | Chunk ID: {chunk['chunk_id']} | Case ID: {chunk['case_id']}")

        print("\nE. FINAL MAXP AGGREGATED TOP 5 UNIQUE CASES:")
        for idx, (chunk, s) in enumerate(hyb_results["final_top_5"], 1):
            print(f"   #{idx} | Score: {s:.4f} | Case ID: {chunk['case_id']} | Best Chunk: {chunk['chunk_id']}")
            print(f"       Evidence: \"{chunk['text'][:120]}...\"")

        audit_results.append(q_audit)

    rep_dir = os.path.abspath(config["paths"]["reports_dir"])
    os.makedirs(rep_dir, exist_ok=True)
    report_json_path = os.path.join(rep_dir, "phase5b_audit_report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)

    print("\n" + "=" * 85)
    print(f"Saved Audit Comparison JSON to '{report_json_path}'")
    print("=" * 85)

if __name__ == "__main__":
    audit()
