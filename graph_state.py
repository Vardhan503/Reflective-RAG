from typing import TypedDict


class GraphState(TypedDict, total=False):
    # Original user request
    question: str

    # Retrieval information
    retrieval_query: str
    retrieval_needed: bool
    router_reason: str

    # Retrieved documents
    documents: list[dict]
    accumulated_documents: list[dict]
    graded_documents: list[dict]

    # CRAG context decision
    crag_route: str
    context_status: str
    context_reason: str
    missing_information: str

    # Generated answer
    answer: str
    last_generated_answer: str
    source_ids: list[str]
    answer_source: str

    # Hallucination checking
    grounded: bool
    hallucination_reason: str
    unsupported_claims: list[str]

    # Answer criticism
    useful: bool
    needs_more_context: bool
    critic_reason: str
    improvement_feedback: str

    # Retry information
    rewrite_count: int
    generation_count: int
    max_retries: int

    # Web fallback
    web_search_used: bool

    # Debugging and evaluation
    failure_reason: str
    elapsed_seconds: float