from typing import TypedDict


class GraphState(TypedDict, total=False):
    question: str
    retrieval_query: str

    retrieval_needed: bool
    router_reason: str

    documents: list[dict]
    graded_documents: list[dict]
    crag_route: str

    answer: str
    source_ids: list[str]

    grounded: bool
    hallucination_reason: str
    unsupported_claims: list[str]

    useful: bool
    critic_reason: str
    improvement_feedback: str

    rewrite_count: int
    generation_count: int
    max_retries: int