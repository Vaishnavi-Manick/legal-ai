import os
import json

proc_path = os.path.join("data", "processed", "chunks.json")
with open(proc_path, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Group chunks by case_id
doc_map = {}
for chk in chunks:
    cid = chk["case_id"]
    if cid not in doc_map:
        doc_map[cid] = []
    doc_map[cid].append(chk)

sample_case_ids = ["judgment_001", "judgment_002", "judgment_003"]

for cid in sample_case_ids:
    chks = doc_map.get(cid, [])
    print("=" * 70)
    print(f"DOCUMENT ID: {cid}")
    if chks:
        first = chks[0]
        print(f"Case Name:  {first['case_name']}")
        print(f"Date:       {first['date']}")
        print(f"Bench:      {first['bench']}")
        print(f"Citation:   {first['citation']}")
        print(f"Source PDF: {first['source_pdf_path']}")
        print(f"Total Chunks: {len(chks)}")
        print("\n--- SAMPLE CHUNK 1 ---")
        print(f"Chunk ID:      {first['chunk_id']}")
        print(f"Section Type:  {first['section_type']}")
        print(f"Page Range:    Page {first['page_start']} to Page {first['page_end']}")
        print(f"Token Count:   {first['token_count']}")
        print(f"Text Snippet:  {first['text'][:300]}...")
    else:
        print("No chunks generated (e.g. Low quality/Scanned PDF).")

