import json
from typing import List, Dict, Any


class LegalRAGPromptBuilder:
    """
    Builds strict, zero-hallucination legal research system and user prompts
    for Grounded Retrieval-Augmented Generation (RAG).
    """

    SYSTEM_PROMPT = (
        "You are a legal research assistant specializing in Indian Supreme Court jurisprudence.\n"
        "Your task is to answer the user's legal query using ONLY the supplied retrieved case evidence.\n\n"
        "STRICT GROUNDING & LEGAL SAFETY RULES:\n"
        "1. Rely ONLY on the facts, principles, and text provided in the RETRIEVED CASE EVIDENCE below.\n"
        "2. Do NOT rely on unsupported external knowledge or invent case facts, holdings, citations, article numbers, legal principles, or dates.\n"
        "3. Distinguish between what the retrieved evidence directly establishes and what cannot be determined from the evidence.\n"
        "4. If the retrieved evidence is insufficient to answer an assertion or the query confidently, explicitly state that the available evidence is insufficient.\n"
        "5. ALWAYS attribute legal statements to their specific source case using the case name and ID (e.g., [Julia Jose Mavely v Union of India / C1208]).\n"
        "6. Conclude your response with this exact concise legal limitation:\n"
        "   \"This output is for legal research assistance and should be independently verified against the full judgments and authoritative legal sources.\"\n\n"
        "STRUCTURE YOUR RESPONSE:\n"
        "### 1. Direct Answer & Legal Overview\n"
        "### 2. Case-by-Case Evidence Synthesis\n"
        "### 3. Key Legal Principles\n"
        "### 4. Limitations & Insufficiency Notes (if applicable)\n"
    )

    @staticmethod
    def build_user_prompt(user_query: str, retrieved_results: List[Dict[str, Any]]) -> str:
        """Constructs formatted user prompt combining user query and retrieved case snippets."""
        evidence_blocks = []

        for idx, item in enumerate(retrieved_results, start=1):
            c_id = item.get("case_id", "Unknown")
            c_name = item.get("case_name") or "Indian Supreme Court Case"
            citation = item.get("citation") or "AILA 2019 Precedent"
            rel_score = item.get("relevance_score", 0.0)
            snippet = item.get("snippet", "").strip()

            block = (
                f"--- RETRIEVED CASE EVIDENCE #{idx} ---\n"
                f"CASE ID: {c_id}\n"
                f"CASE NAME: {c_name}\n"
                f"RELEVANCE SCORE: {rel_score:.4f}\n"
                f"CITATION: {citation}\n"
                f"EVIDENCE SNIPPET:\n\"{snippet}\"\n"
            )
            evidence_blocks.append(block)

        evidence_text = "\n".join(evidence_blocks)

        user_prompt = (
            f"USER QUERY:\n{user_query.strip()}\n\n"
            f"RETRIEVED CASES (Top {len(retrieved_results)} Evidence Items):\n"
            f"{evidence_text}\n\n"
            f"INSTRUCTIONS:\n"
            f"Provide a concise, structured, evidence-grounded legal answer addressing the user's query.\n"
            f"Attribute every legal statement directly to its supporting case (e.g., [{retrieved_results[0].get('case_name', 'Case')} / {retrieved_results[0].get('case_id', 'C1')}])."
        )

        return user_prompt
