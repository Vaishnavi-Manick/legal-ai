import re
import logging
from typing import Dict, Any

logger = logging.getLogger("LegalRAG.Retrieval.Router")


class CitationRouter:
    """Detects whether a query is a legal citation or case-name lookup to bypass query rewriting."""

    CITATION_PATTERNS = [
        re.compile(r"\b\d{4}\s+SCC\s+\d+\b", re.IGNORECASE),
        re.compile(r"\b\d{4}\s+AIR\s+\d+\b", re.IGNORECASE),
        re.compile(r"\bWrit\s+Petition\s+No\.?\s*\d+", re.IGNORECASE),
        re.compile(r"\bCivil\s+Appeal\s+No\.?\s*\d+", re.IGNORECASE),
        re.compile(r"\bCriminal\s+Appeal\s+No\.?\s*\d+", re.IGNORECASE),
        re.compile(r"\b[A-Z][a-z0-9.]+\s+(?:v\.|vs\.|versus)\s+[A-Z][a-z0-9.]+\b", re.IGNORECASE),
    ]

    def route_query(self, query: str) -> Dict[str, Any]:
        cleaned = query.strip()

        is_citation_or_case = any(pattern.search(cleaned) for pattern in self.CITATION_PATTERNS)

        if is_citation_or_case:
            logger.info(f"Query Router: Detected citation or case-name pattern in '{query}'. Skipping query rewriting.")
            return {
                "is_citation": True,
                "skip_rewriting": True,
                "query_type": "citation_or_case_lookup",
                "query": cleaned
            }
        else:
            return {
                "is_citation": False,
                "skip_rewriting": False,
                "query_type": "general_legal_query",
                "query": cleaned
            }
