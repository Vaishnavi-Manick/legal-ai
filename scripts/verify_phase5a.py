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
        "phase": "5A - Case-Level MaxP Aggregation Verification",
        "queries": []
    }

    print("=" * 85)
    print("      PHASE 5A CASE-LEVEL RESULT AGGREGATION & RETRIEVAL VERIFICATION")
    print("=" * 85)

    for q in test_queries:
        print(f"\n[QUERY] '{q}'")
        res = engine.search(q)

        q_report = {
            "query": res["original_query"],
            "duplicates_removed_flag": res.get("has_duplicate_cases_removed", False),
            "duplicates_removed_count": res.get("duplicates_removed_count", 0),
            "total_latency": res["latencies"]["total_time"],
            "unique_top_5_cases": []
        }

        print(f"  - Original Query:            '{res['original_query']}'")
        print(f"  - Duplicate Cases Removed:   {'[YES] (' + str(res.get('duplicates_removed_count', 0)) + ' duplicate chunks removed)' if res.get('has_duplicate_cases_removed') else '[NO]'}")
        print(f"  - Total Latency:             {res['latencies']['total_time']:.4f}s")
        print(f"  - Fallback Triggered:        {'YES' if res['used_fallback'] else 'NO'}")

        print("\n  --- TOP 5 UNIQUE CASE-LEVEL RESULTS (Sorted by Best Chunk Score MaxP) ---")
        unique_case_ids = []

        for rank, (chunk, score) in enumerate(res["final_top_5"], 1):
            case_id = chunk["case_id"]
            unique_case_ids.append(case_id)

            item = {
                "rank": rank,
                "case_id": case_id,
                "case_name": chunk.get("case_name", "Unknown Case"),
                "best_chunk_score": round(score, 4),
                "best_chunk_id": chunk["chunk_id"],
                "section_type": chunk.get("section_type"),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "source_pdf_path": chunk.get("source_pdf_path"),
                "evidence_snippet": chunk["text"][:250] + "..."
            }
            q_report["unique_top_5_cases"].append(item)

            print(f"   #{rank} | MaxP Score: {item['best_chunk_score']:.4f} | Unique Case ID: {item['case_id']}")
            print(f"       Case Title:  {item['case_name']}")
            print(f"       Best Chunk:  {item['best_chunk_id']} | Pages: Page {item['page_start']} to {item['page_end']}")
            print(f"       PDF Path:    {item['source_pdf_path']}")
            print(f"       Evidence:    \"{item['evidence_snippet']}\"")
            print()

        # Sanity check for uniqueness
        assert len(unique_case_ids) == len(set(unique_case_ids)), f"Duplicate case IDs found in Top 5 for query '{q}'!"
        print(f"  [PASS] SANITY CHECK PASSED: All {len(unique_case_ids)} returned cases are 100% UNIQUE.")

        report["queries"].append(q_report)

    # Save Phase 5A report JSON
    rep_dir = os.path.abspath(config["paths"]["reports_dir"])
    os.makedirs(rep_dir, exist_ok=True)
    report_json_path = os.path.join(rep_dir, "phase5a_verification_report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 85)
    print(f"Saved Phase 5A Verification Report to '{report_json_path}'")
    print("=" * 85)


if __name__ == "__main__":
    main()
