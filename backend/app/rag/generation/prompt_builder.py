from typing import Any, Dict, List


class PromptBuilder:
    """
    Constructs strict, grounded system instructions and context-augmented
    user prompts enforcing verifiable source attribution and zero-hallucination guardrails.
    """

    SYSTEM_PROMPT = (
        "You are SIFT, an enterprise document intelligence and semantic reasoning assistant.\n"
        "Your mission is to provide accurate, truthful, and strictly grounded answers using ONLY "
        "the provided document excerpts.\n\n"
        "MANDATORY INSTRUCTIONS:\n"
        "1. GROUNDING: Base your answer strictly on the facts, specifications, and evidence explicitly "
        "stated in the provided excerpts below. Do not use outside knowledge or extrapolate beyond the text.\n"
        "2. INSUFFICIENT EVIDENCE: If the provided excerpts do not contain enough factual evidence to answer "
        "the question completely, you MUST state explicitly:\n"
        "   \"Based on the provided documents, there is insufficient evidence to answer your question.\"\n"
        "3. CITATIONS: Attribute factual claims to the corresponding excerpt using bracket numbers, "
        "e.g. [1], [2], corresponding to the passage index.\n"
        "4. HONESTY: Never invent document names, page numbers, quotations, or metrics. "
        "Distinguish direct evidence from reasonable interpretation.\n"
        "5. TONE: Professional, objective, clear, and direct."
    )

    @classmethod
    def build_grounded_prompt(
        cls,
        query: str,
        retrieved_passages: List[Dict[str, Any]],
    ) -> str:
        """
        Builds the context-augmented user prompt containing retrieved document passages
        and the natural-language question.
        """
        if not retrieved_passages:
            return (
                "CONTEXT EXCERPTS:\n"
                "[NO RELEVANT DOCUMENT CONTEXT AVAILABLE]\n\n"
                f"QUESTION: {query}\n\n"
                "INSTRUCTION: Since no relevant document passages were retrieved, state that there is "
                "insufficient evidence to answer the question based on the provided documents."
            )

        context_blocks = []
        for idx, p in enumerate(retrieved_passages, start=1):
            source = p.get("source_filename", "Document")
            page_start = p.get("page_start")
            page_end = p.get("page_end")

            if page_start is not None and page_end is not None:
                if page_start == page_end:
                    page_str = f"Page {page_start}"
                else:
                    page_str = f"Pages {page_start}-{page_end}"
            elif page_start is not None:
                page_str = f"Page {page_start}"
            else:
                page_str = "Page N/A"

            chunk_id = p.get("chunk_id", f"chk_{idx}")
            content = (p.get("content") or "").strip()

            block = (
                f"--- Passage [{idx}] ---\n"
                f"Source: {source} ({page_str}) [ID: {chunk_id}]\n"
                f"Content: {content}"
            )
            context_blocks.append(block)

        formatted_context = "\n\n".join(context_blocks)

        return (
            "CONTEXT EXCERPTS FROM RETRIEVED DOCUMENTS:\n"
            f"{formatted_context}\n\n"
            f"USER QUESTION: {query}\n\n"
            "ANSWER (grounded strictly in the excerpts above, citing sources using [1], [2], etc.):"
        )
