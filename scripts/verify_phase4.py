import os
import sys
import json
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ingest.pipeline import load_config
from retrieval.query_engine import Phase3QueryEngine

def main():
    config = load_config("config.yaml")
    engine = Phase3QueryEngine(config)

    test_queries = [
        "right to privacy under Article 21",
        "constitutional validity of preventive detention",
        "principles of natural justice"
    ]

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "queries": []
    }

    print("=" * 80)
    print("         PHASE 4 END-TO-END PIPELINE VERIFICATION REPORT")
    print("=" * 80)

    for q in test_queries:
        print(f"\n[QUERY] '{q}'")
        res = engine.search(q)

        bm25_ok = len(res["bm25_candidates"]) > 0
        dense_ok = len(res["dense_candidates"]) > 0

        q_report = {
            "original_query": res["original_query"],
            "rewritten_query": res["rewritten_query"],
            "was_rewritten": res["was_rewritten"],
            "rewrite_reason": res["rewrite_reason"],
            "is_citation": res["is_citation"],
            "bm25_succeeded": bm25_ok,
            "dense_succeeded": dense_ok,
            "used_fallback": res["used_fallback"],
            "fallback_reason": res["fallback_reason"],
            "latencies": res["latencies"],
            "bm25_top": [{"chunk_id": c[0]["chunk_id"], "case_id": c[0]["case_id"], "score": round(c[1], 4)} for c in res["bm25_candidates"]],
            "dense_top": [{"chunk_id": c[0]["chunk_id"], "case_id": c[0]["case_id"], "score": round(c[1], 4)} for c in res["dense_candidates"]],
            "rrf_top": [{"chunk_id": c[0]["chunk_id"], "case_id": c[0]["case_id"], "score": round(c[1], 4)} for c in res["rrf_candidates"]],
            "final_top_5": []
        }

        print(f"  - Original Query:    '{res['original_query']}'")
        print(f"  - Rewritten Query:   '{res['rewritten_query']}' (Rewritten: {res['was_rewritten']}, Reason: {res['rewrite_reason']})")
        print(f"  - Citation Route:    {res['is_citation']}")
        print(f"  - BM25 Status:       {'[SUCCESS]' if bm25_ok else '[FAILED]'}")
        print(f"  - Dense Status:      {'[SUCCESS]' if dense_ok else '[FAILED]'}")
        print(f"  - Fallback Triggered: {'YES' if res['used_fallback'] else 'NO'}")
        print(f"  - Stage Latencies:   {res['latencies']}")

        print("\n  --- TRANSFORMER V1 RERANKED TOP 5 RESULTS ---")
        for rank, (chunk, score) in enumerate(res["final_top_5"], 1):
            item = {
                "rank": rank,
                "score": round(score, 4),
                "chunk_id": chunk["chunk_id"],
                "case_id": chunk["case_id"],
                "case_name": chunk.get("case_name", "Unknown Case"),
                "date": chunk.get("date"),
                "bench": chunk.get("bench"),
                "citation": chunk.get("citation"),
                "section_type": chunk.get("section_type"),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "source_pdf_path": chunk.get("source_pdf_path"),
                "evidence_snippet": chunk["text"][:250] + "..."
            }
            q_report["final_top_5"].append(item)

            print(f"   #{rank} | Score: {item['score']:.4f} | Chunk ID: {item['chunk_id']} | Case ID: {item['case_id']}")
            print(f"       Case Title:  {item['case_name']}")
            print(f"       Citation:    {item['citation']} | Pages: Page {item['page_start']} to {item['page_end']}")
            print(f"       PDF Path:    {item['source_pdf_path']}")
            print(f"       Evidence:    \"{item['evidence_snippet']}\"")
            print()

        report["queries"].append(q_report)

    # Test Fallback Functionality explicitly
    print("\n" + "=" * 80)
    print("         TESTING V1 FALLBACK MECHANISM (Simulated Fault)")
    print("=" * 80)
    print("Triggering intentional exception in hybrid retrieval to test V1 fallback...")
    
    real_indexer = engine.dense_indexer
    engine.dense_indexer = None  # Force exception
    fb_test_res = engine.search("principles of natural justice")
    engine.dense_indexer = real_indexer  # Restore

    print(f"  - Fallback Triggered: {'[YES]' if fb_test_res['used_fallback'] else '[NO]'}")
    print(f"  - Fallback Reason:    {fb_test_res['fallback_reason']}")
    print(f"  - Fallback Results:   {len(fb_test_res['final_top_5'])} results returned via V1 fallback.")

    # Save verification report JSON
    rep_dir = os.path.abspath(config["paths"]["reports_dir"])
    os.makedirs(rep_dir, exist_ok=True)
    report_json_path = os.path.join(rep_dir, "phase4_verification_report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Saved Phase 4 Verification Report to '{report_json_path}'")
    print("=" * 80)

if __name__ == "__main__":
    main()

