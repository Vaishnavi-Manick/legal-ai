import os
import sys
import argparse
import logging
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ingest.pipeline import IngestionPipeline, load_config
from index.pipeline import IndexingPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("LegalRAG.CLI")


def main():
    parser = argparse.ArgumentParser(description="Legal RAG Multi-Phase CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: ingest (Phase 1)
    ingest_parser = subparsers.add_parser("ingest", help="Run Phase 1 PDF detection, extraction, cleaning, parsing & chunking")
    ingest_parser.add_argument("--limit", type=int, default=50, help="Maximum number of PDFs to process")
    ingest_parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")

    # Command: index (Phase 2)
    index_parser = subparsers.add_parser("index", help="Run Phase 2 Dense & BM25 indexing over processed chunks")
    index_parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")

    # Command: ask (placeholder for Phase 4+)
    ask_parser = subparsers.add_parser("ask", help="Query the Legal RAG system (Phase 4+)")
    ask_parser.add_argument("query", type=str, help="Legal question or case lookup query")

    args = parser.parse_args()

    if args.command == "ingest":
        config = load_config(args.config)
        pdf_dir = os.path.abspath(config["paths"]["sample_pdf_dir"])
        logger.info(f"Starting Phase 1 Ingestion on directory '{pdf_dir}' (Limit: {args.limit})...")
        pipeline = IngestionPipeline(config)
        results = pipeline.run_on_directory(pdf_dir=pdf_dir, limit=args.limit)

        print("\n" + "=" * 70)
        print("                  PHASE 1 INGESTION PIPELINE COMPLETE")
        print("=" * 70)
        print(f"Total PDFs Processed:  {results['report']['total_documents_processed']}")
        print(f"Native PDFs:           {results['report']['native_pdfs_count']}")
        print(f"Scanned PDFs:          {results['report']['scanned_pdfs_count']}")
        print(f"Low-Quality Flagged:   {results['report']['low_quality_flagged_count']}")
        print(f"Total Chunks:          {results['report']['total_chunks_generated']}")
        print(f"Artifacts Saved To:    {os.path.abspath(config['paths']['processed_dir'])}")
        print(f"OCR Quality Report:    {os.path.join(os.path.abspath(config['paths']['reports_dir']), 'ocr_quality_report.md')}")
        print("=" * 70)

    elif args.command == "index":
        config = load_config(args.config)
        logger.info("Starting Phase 2 Dense & BM25 Indexing Pipeline...")
        indexer_pipeline = IndexingPipeline(config)
        summary = indexer_pipeline.run_indexing()

        print("\n" + "=" * 70)
        print("                  PHASE 2 INDEXING PIPELINE COMPLETE")
        print("=" * 70)
        print(f"Total Chunks Processed:   {summary['total_chunks']}")
        print(f"Chunks Indexed:           {summary['indexed_successfully']}")
        print(f"Indexing Failures:        {summary['indexing_failures']}")
        print(f"1-to-1 ID Alignment:      {'VERIFIED (1-to-1)' if summary['alignment_verified'] else 'MISMATCH'}")
        print(f"Dense Embedding Model:    {summary['dense_model']} ({summary['embedding_dim']}d)")
        print(f"BM25 Index Algorithm:     {summary['bm25_algorithm']}")
        print(f"Vector Store Dir:         {os.path.abspath(config['paths']['vector_db_dir'])}")
        print(f"BM25 Index Dir:           {os.path.abspath(config['paths']['bm25_index_dir'])}")
        print("=" * 70)

    elif args.command == "ask":
        print(f"Query path will be executed in Phase 4 for: '{args.query}'")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
