from typing import TypedDict


class GraphState(TypedDict, total=False):
    question: str # User Question
    retrieval_query: str # current search query

    retrieval_needed: bool # Self-RAG Retrieval Needed
    router_reason: str # Self-RAG Router Reason

    documents: list[dict] # Retrieved Documents
    graded_documents: list[dict] # Documents + correct/ambiguous/incorrect labels
    crag_route: str # Selects Generation route

    answer: str # Generated Answer checking for hallucination and usefulness
    source_ids: list[str] # Retrieved Document IDs cited by generated answer

    grounded: bool # Hallucination Checker Result
    hallucination_reason: str # Hallucination Checker Reason
    unsupported_claims: list[str] # Hallucination Checker Unsupported Claims

    useful: bool # Answer Critic Result
    critic_reason: str # Answer Critic Reason
    improvement_feedback: str # Answer Critic Improvement Feedback, fed back to answer generator

    rewrite_count: int # Rewrite Count
    generation_count: int # Generation Count
    max_retries: int # Max Retries