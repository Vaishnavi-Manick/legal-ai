import os
import sys
import json
import time
import logging

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ingest.pipeline import load_config
from retrieval.query_engine import Phase3QueryEngine
from src.inference import LegalSearchEngine

logger = logging.getLogger("LegalRAG.Scripts.VerifyPhase5C")

LANDMARK_CASES = {
    "right to privacy under Article 21": ["C2643", "C2411", "C1902", "C2814"],
    "constitutional validity of preventive detention": ["C2249", "C1235", "C810", "C1111", "C1332"],
    "principles of natural justice": ["C2263", "C843", "C1808", "C2473", "C660"]
}


def main():
    config = load_config("config.yaml")
    print("=" * 85)
    print("      PHASE 5C FULL AILA 2019 INDEX & V1-COMPATIBLE CHUNKING VERIFICATION")
    print("=" * 85)

    print("\n[1/2] Initializing Full-Corpus Hybrid Query Engine (Phase 3 Pipeline)...")
    hybrid_engine = Phase3QueryEngine(config)

    print("\n[2/2] Initializing Original V1 Search Engine (Baseline)...")
    v1_engine = LegalSearchEngine(candidate_pool_size=50)

    test_queries = [
        "right to privacy under Article 21",
        "constitutional validity of preventive detention",
        "principles of natural justice"
    ]

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "phase": "5C - Full Corpus Index + V1-Compatible Head Chunking",
        "corpus_docs": 2914,
        "queries": []
    }

    for q in test_queries:
        print("\n" + "=" * 85)
        print(f"VERIFYING QUERY: '{q}'")
        print("=" * 85)

        # 1. Execute Full Corpus Hybrid Search Engine
        res = hybrid_engine.search(q)

        # 2. Execute Original V1 Engine
        v1_res = v1_engine.search(q, top_k=5)

        # 3. Retrieve BM25 & Dense Top 10 for inspection
        bm25_top_10 = hybrid_engine.bm25_indexer.search(res["rewritten_query"], top_k=10)
        dense_top_10 = hybrid_engine.dense_indexer.search(res["rewritten_query"], top_k=10)

        bm25_succeeded = len(bm25_top_10) > 0
        dense_succeeded = len(dense_top_10) > 0

        # Check landmark cases in Hybrid Top 5 and Original V1 Top 5
        target_landmarks = LANDMARK_CASES.get(q, [])
        hybrid_top_cases = [c[0]["case_id"] for c in res["final_top_5"]]
        v1_top_cases = [r["case_id"] for r in v1_res]

        hybrid_landmarks_found = [cid for cid in target_landmarks if cid in hybrid_top_cases]
        v1_landmarks_found = [cid for cid in target_landmarks if cid in v1_top_cases]

        # Detailed Query Report
        q_report = {
            "original_query": res["original_query"],
            "rewritten_query": res["rewritten_query"],
            "was_rewritten": res.get("was_rewritten", False),
            "rewrite_reason": res.get("rewrite_reason"),
            "bm25_succeeded": bm25_succeeded,
            "dense_succeeded": dense_succeeded,
            "bm25_top_10": [
                {"rank": i + 1, "chunk_id": c[0]["chunk_id"], "case_id": c[0]["case_id"], "score": round(c[1], 4)}
                for i, c in enumerate(bm25_top_10)
            ],
            "dense_top_10": [
                {"rank": i + 1, "chunk_id": c[0]["chunk_id"], "case_id": c[0]["case_id"], "score": round(c[1], 4)}
                for i, c in enumerate(dense_top_10)
            ],
            "rrf_top_10": [
                {"rank": i + 1, "chunk_id": c[0]["chunk_id"], "case_id": c[0]["case_id"], "rrf_score": round(c[1], 6)}
                for i, c in enumerate(res["rrf_candidates"][:10])
            ],
            "hybrid_v1_top_5": [],
            "v1_baseline_top_5": [
                {
                    "rank": r["rank"],
                    "case_id": r["case_id"],
                    "case_name": r.get("case_name", "Unknown Case"),
                    "score": round(r["relevance_score"], 4),
                    "snippet": r["snippet"][:200]
                }
                for r in v1_res
            ],
            "landmarks_checked": target_landmarks,
            "hybrid_landmarks_in_top5": hybrid_landmarks_found,
            "v1_landmarks_in_top5": v1_landmarks_found,
            "latencies": res["latencies"],
            "used_fallback": res["used_fallback"],
            "fallback_reason": res["fallback_reason"]
        }

        print(f"  - Original Query:       '{res['original_query']}'")
        print(f"  - Rewritten Query:      '{res['rewritten_query']}' (Rewritten: {res['was_rewritten']})")
        print(f"  - BM25 Retrieval:       {'SUCCESS' if bm25_succeeded else 'FAILED'} ({len(bm25_top_10)} top candidates loaded)")
        print(f"  - Dense Retrieval:      {'SUCCESS' if dense_succeeded else 'FAILED'} ({len(dense_top_10)} top candidates loaded)")
        print(f"  - Total Pipeline Time:  {res['latencies']['total_time']:.4f}s")
        print(f"  - Fallback Triggered:   {'YES' if res['used_fallback'] else 'NO'}")

        print("\n  --- TOP 5 HYBRID + TRANSFORMER V1 UNIQUE CASE RESULTS ---")
        unique_case_ids = []
        for rank, (chunk, score) in enumerate(res["final_top_5"], 1):
            case_id = chunk["case_id"]
            unique_case_ids.append(case_id)
            snippet = chunk["text"][:220] + "..." if len(chunk["text"]) > 220 else chunk["text"]

            item = {
                "rank": rank,
                "case_id": case_id,
                "case_name": chunk.get("case_name", "Unknown Case"),
                "maxp_score": round(score, 4),
                "best_chunk_id": chunk["chunk_id"],
                "section_type": chunk.get("section_type"),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "source_pdf_path": chunk.get("source_pdf_path"),
                "evidence_snippet": snippet
            }
            q_report["hybrid_v1_top_5"].append(item)

            print(f"   #{rank} | MaxP Score: {item['maxp_score']:.4f} | Case ID: {item['case_id']}")
            print(f"       Case Title:  {item['case_name']}")
            print(f"       Chunk ID:    {item['best_chunk_id']} | Pages: {item['page_start']}-{item['page_end']}")
            print(f"       Evidence:    \"{snippet}\"")
            print()

        assert len(unique_case_ids) == len(set(unique_case_ids)), f"Duplicate case IDs found in Top 5 for query '{q}'!"
        print(f"  [PASS] All {len(unique_case_ids)} returned cases are 100% UNIQUE.")

        print("\n  --- LANDMARK CASE EVALUATION ---")
        print(f"   Target Landmark Cases:  {target_landmarks}")
        print(f"   Hybrid+V1 Top 5 Found:   {hybrid_landmarks_found}")
        print(f"   Original V1 Top 5 Found: {v1_landmarks_found}")

        print("\n  --- BASELINE COMPARISON (Original V1 TF-IDF Engine) ---")
        for r in v1_res:
            print(f"   #{r['rank']} | Score: {r['relevance_score']:.4f} | Case ID: {r['case_id']} | Title: {r.get('case_name', 'Unknown')}")
            print(f"       Snippet: \"{r['snippet'][:120]}...\"")

        report["queries"].append(q_report)

    # Save verification report
    rep_dir = os.path.abspath(config["paths"]["reports_dir"])
    os.makedirs(rep_dir, exist_ok=True)
    report_json_path = os.path.join(rep_dir, "phase5c_verification_report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 85)
    print(f"Saved Phase 5C Full Verification Report to '{report_json_path}'")
    print("=" * 85)


if __name__ == "__main__":
    main()

