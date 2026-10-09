import json
import time
import logging
from typing import Dict, Any, List, Optional
from src.inference import LegalSearchEngine
from src.llm.llm_client import LLMClient
from src.llm.prompt_builder import LegalRAGPromptBuilder

logger = logging.getLogger("LegalRAGPipeline")


class LegalRAGPipeline:
    """
    Phase 2 RAG Pipeline:
    Grounded LLM Answer Generation Layer ON TOP OF Verified V1 Transformer Search Engine.
    
    Pipeline Steps:
      1. User Query
      2. Existing V1 Retrieval (TF-IDF -> Transformer V1 Cross-Encoder Reranker) -> Top 5
      3. Evidence / Passage Extraction from source cases
      4. Grounded Prompt Construction
      5. Configurable LLM Client (OpenAI / Custom / Fallback)
      6. Structured Response Generation
    """
    def __init__(self, search_engine: Optional[LegalSearchEngine] = None):
        if search_engine is None:
            logger.info("Initializing default V1 LegalSearchEngine for RAG pipeline...")
            search_engine = LegalSearchEngine(candidate_pool_size=50)

        self.search_engine = search_engine
        self.llm_client = LLMClient()
        self.prompt_builder = LegalRAGPromptBuilder()

    def run_pipeline(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Executes end-to-end RAG workflow.
        Guarantees that retrieved cases are ALWAYS returned, even if LLM generation is unavailable or fails.
        """
        start_time = time.time()
        query_text = (query_text or "").strip()

        if not query_text:
            return {
                "query": query_text,
                "llm_available": False,
                "answer": None,
                "notice": "Query cannot be empty.",
                "supporting_cases": [],
                "results": [],
                "latency": {"retrieval_time": 0.0, "llm_time": 0.0, "total_time": 0.0}
            }

        # Step 1 & 2: Execute Verified V1 Retrieval
        start_retrieval = time.time()
        try:
            results = self.search_engine.search(query_text=query_text, top_k=top_k)
        except Exception as e:
            logger.error(f"V1 Retrieval failed: {str(e)}", exc_info=True)
            results = []

        retrieval_time = round(time.time() - start_retrieval, 3)

        # Step 3: Extract Supporting Cases & Real Evidence Snippets
        supporting_cases = []
        for r in results:
            supporting_cases.append({
                "case_id": r["case_id"],
                "case_name": r.get("case_name") or f"Case {r['case_id']}",
                "citation": r.get("citation") or "AILA 2019 Precedent",
                "evidence": r["snippet"],
                "relevance_score": r["relevance_score"],
                "document_text": r["document_text"]
            })

        # Default fallback response structure if LLM is unavailable or fails
        fallback_notice = "LLM answer generation is unavailable. The retrieved cases and supporting evidence are shown below."
        disclaimer = "Research assistance only. Verify against the authoritative judgment before relying on it."

        # Step 4: Check if LLM is available
        if not self.llm_client.is_available() or not results:
            total_time = round(time.time() - start_time, 3)
            return {
                "query": query_text,
                "llm_available": False,
                "answer": None,
                "notice": fallback_notice if results else "No relevant legal cases found for this query.",
                "supporting_cases": supporting_cases,
                "results": results,
                "disclaimer": disclaimer,
                "confidence_note": "Retrieved cases shown directly without LLM synthesis.",
                "latency": {
                    "retrieval_time": retrieval_time,
                    "llm_time": 0.0,
                    "total_time": total_time
                }
            }

        # Step 5: Build Prompt & Invoke LLM
        start_llm = time.time()
        system_prompt = self.prompt_builder.SYSTEM_PROMPT
        user_prompt = self.prompt_builder.build_user_prompt(query_text, results)

        raw_llm_output = self.llm_client.generate_response(system_prompt, user_prompt, temperature=0.0)
        llm_time = round(time.time() - start_llm, 3)
        total_time = round(time.time() - start_time, 3)

        if not raw_llm_output:
            logger.warning("LLM call returned None/failed. Falling back to retrieved case list.")
            return {
                "query": query_text,
                "llm_available": False,
                "answer": None,
                "notice": fallback_notice,
                "supporting_cases": supporting_cases,
                "results": results,
                "disclaimer": disclaimer,
                "confidence_note": "LLM call failed/timed out. Showing retrieved cases directly.",
                "latency": {
                    "retrieval_time": retrieval_time,
                    "llm_time": llm_time,
                    "total_time": total_time
                }
            }

        # Step 6: Parse LLM Output (JSON or raw text)
        parsed_answer = None
        confidence_note = "High confidence grounded in retrieved AILA precedents."

        try:
            # Try parsing JSON output
            clean_json_str = raw_llm_output
            if "```json" in clean_json_str:
                clean_json_str = clean_json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_json_str:
                clean_json_str = clean_json_str.split("```")[1].split("```")[0].strip()

            parsed_data = json.loads(clean_json_str)
            if isinstance(parsed_data, dict):
                parsed_answer = parsed_data.get("answer")
                confidence_note = parsed_data.get("confidence_note", confidence_note)
                disclaimer = parsed_data.get("disclaimer", disclaimer)
        except Exception:
            logger.info("Raw LLM output was not strict JSON. Using plain text answer.")
            parsed_answer = raw_llm_output

        return {
            "query": query_text,
            "llm_available": True,
            "answer": parsed_answer,
            "notice": None,
            "supporting_cases": supporting_cases,
            "results": results,
            "confidence_note": confidence_note,
            "disclaimer": disclaimer,
            "latency": {
                "retrieval_time": retrieval_time,
                "llm_time": llm_time,
                "total_time": total_time
            }
        }

