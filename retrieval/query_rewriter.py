import logging
import requests
from typing import Dict, Any

logger = logging.getLogger("LegalRAG.Retrieval.QueryRewriter")


class LLMQueryRewriter:
    """Legal Query Rewriter using local Ollama at temp 0 with safe fallbacks."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        env_cfg = config.get("environment", {})
        q_cfg = config.get("query_pipeline", {})

        self.enabled = q_cfg.get("query_rewrite_enabled", False)
        self.base_url = env_cfg.get("ollama_base_url", "http://localhost:11434")
        self.model = env_cfg.get("llm_model", "qwen2.5:3b")

    def rewrite_query(self, original_query: str, skip_rewriting: bool = False) -> Dict[str, Any]:
        result = {
            "original_query": original_query,
            "rewritten_query": original_query,
            "was_rewritten": False,
            "reason": "disabled_or_skipped"
        }

        if skip_rewriting:
            result["reason"] = "skipped_by_citation_router"
            logger.info(f"QueryRewriter: Skipped rewriting because query is a citation/case-name.")
            return result

        if not self.enabled:
            result["reason"] = "query_rewrite_disabled_in_config"
            logger.info(f"QueryRewriter: Query rewrite disabled in config. Using original query.")
            return result

        # Attempt call to local Ollama API
        prompt = f"""You are an expert Indian Legal AI Assistant. Rewrite the user's legal research query to expand legal synonyms and concepts while STRICTLY preserving all Article numbers, Section numbers, Act names, and Case names. Do NOT alter any numbers or case names.

User Query: "{original_query}"

Return ONLY the single rewritten query text, with no explanations or preamble."""

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0}
                },
                timeout=5.0
            )

            if resp.status_code == 200:
                data = resp.json()
                rewritten_text = data.get("response", "").strip().replace('"', '')
                if rewritten_text:
                    result["rewritten_query"] = rewritten_text
                    result["was_rewritten"] = True
                    result["reason"] = "successfully_rewritten"
                    logger.info(f"QueryRewriter: Rewrote '{original_query}' -> '{rewritten_text}'")
                    return result

        except Exception as e:
            logger.warning(f"QueryRewriter: Ollama LLM unavailable or timed out ({e}). Falling back to original query.")

        result["reason"] = "llm_unavailable_fallback"
        return result

