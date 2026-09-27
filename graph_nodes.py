from answer_critic import critique_answer
from answer_generator import generate_answer
from crag_context_grader import assess_context
from crag_document_grader import grade_documents
from graph_state import GraphState
from hallucination_checker import check_hallucination
from hybrid_retrieval import hybrid_retrieve
from models import generator_model
from query_rewriter import rewrite_query
from retrieval_router import decide_if_retrieval_is_needed
from web_search import search_web


def merge_documents(existing_documents, new_documents):
    merged_documents = []
    seen_document_ids = set()

    for document in existing_documents:
        document_id = document["id"]

        if document_id not in seen_document_ids:
            merged_documents.append(document)
            seen_document_ids.add(document_id)

    for document in new_documents:
        document_id = document["id"]

        if document_id not in seen_document_ids:
            merged_documents.append(document)
            seen_document_ids.add(document_id)

    return merged_documents


def retrieval_router_node(state: GraphState) -> GraphState:
    question = state["question"]

    decision = decide_if_retrieval_is_needed(question)

    return {
        "retrieval_needed": decision.retrieve,
        "router_reason": decision.reason,
    }


def direct_response_node(state: GraphState) -> GraphState:
    question = state["question"]

    prompt = f"""
Respond directly to the user's request.

This request does not require retrieval from external documents.

Do not claim that you searched a knowledge base or the web.

User request:
{question}
"""

    response = generator_model.invoke(prompt)

    return {
        "answer": response.content,
        "last_generated_answer": response.content,
        "source_ids": [],
        "grounded": True,
        "useful": True,
        "needs_more_context": False,
        "answer_source": "direct",
    }


def retrieve_documents_node(state: GraphState) -> GraphState:
    retrieval_query = state.get(
        "retrieval_query",
        state["question"],
    )

    new_documents = hybrid_retrieve(retrieval_query)

    existing_documents = state.get(
        "accumulated_documents",
        [],
    )

    combined_documents = merge_documents(
        existing_documents=existing_documents,
        new_documents=new_documents,
    )

    return {
        "retrieval_query": retrieval_query,
        "documents": combined_documents,
        "accumulated_documents": combined_documents,
    }


def grade_documents_node(state: GraphState) -> GraphState:
    question = state["question"]
    documents = state["documents"]

    graded_documents = grade_documents(
        question=question,
        documents=documents,
    )

    relevant_documents = []

    for document in graded_documents:
        if document["grade"] != "incorrect":
            relevant_documents.append(document)

    context_assessment = assess_context(
        question=question,
        documents=relevant_documents,
    )

    if context_assessment.status == "sufficient":
        crag_route = "generate"

    elif context_assessment.status == "incomplete":
        crag_route = "rewrite_query"

    else:
        crag_route = "search_web"

    return {
        "documents": relevant_documents,
        "accumulated_documents": relevant_documents,
        "graded_documents": graded_documents,
        "context_status": context_assessment.status,
        "context_reason": context_assessment.reason,
        "missing_information": (
            context_assessment.missing_information
        ),
        "crag_route": crag_route,
    }


def rewrite_query_node(state: GraphState) -> GraphState:
    question = state["question"]

    current_query = state.get(
        "retrieval_query",
        question,
    )

    documents = state.get("documents", [])

    missing_information = state.get(
        "missing_information",
        "",
    )

    rewrite_result = rewrite_query(
        question=question,
        current_query=current_query,
        documents=documents,
        missing_information=missing_information,
    )

    current_rewrite_count = state.get(
        "rewrite_count",
        0,
    )

    return {
        "retrieval_query": rewrite_result.rewritten_query,
        "rewrite_count": current_rewrite_count + 1,

        # New documents should receive fresh generation attempts.
        "generation_count": 0,

        # Remove feedback created for the previous context.
        "improvement_feedback": "",
        "unsupported_claims": [],

        "answer_source": "documents",
    }


def web_search_node(state: GraphState) -> GraphState:
    search_query = state.get(
        "retrieval_query",
        state["question"],
    )

    web_documents = search_web(
        search_query,
        max_results=5,
    )

    existing_documents = state.get(
        "accumulated_documents",
        [],
    )

    combined_documents = merge_documents(
        existing_documents=existing_documents,
        new_documents=web_documents,
    )

    return {
        "documents": combined_documents,
        "accumulated_documents": combined_documents,
        "graded_documents": [],
        "crag_route": "generate",
        "context_status": "web",
        "context_reason": (
            "Local context was incomplete. "
            "Web search results were added."
        ),
        "missing_information": "",
        "answer_source": "web",
        "web_search_used": True,
        "generation_count": 0,
        "improvement_feedback": "",
        "unsupported_claims": [],
    }


def generate_answer_node(state: GraphState) -> GraphState:
    question = state["question"]
    documents = state["documents"]

    generation_question = question

    improvement_feedback = state.get(
        "improvement_feedback",
        "",
    )

    if improvement_feedback != "":
        generation_question = generation_question + "\n\n"
        generation_question = generation_question + (
            "Regeneration instructions: "
        )
        generation_question = (
            generation_question + improvement_feedback
        )

    unsupported_claims = state.get(
        "unsupported_claims",
        [],
    )

    if len(unsupported_claims) > 0:
        generation_question = generation_question + "\n\n"
        generation_question = generation_question + (
            "Do not repeat these unsupported claims:"
        )

        for claim in unsupported_claims:
            generation_question = generation_question + "\n- "
            generation_question = generation_question + claim

    generated_result = generate_answer(
        question=generation_question,
        documents=documents,
    )

    current_generation_count = state.get(
        "generation_count",
        0,
    )

    return {
        "answer": generated_result.answer,
        "last_generated_answer": generated_result.answer,
        "source_ids": generated_result.source_ids,
        "generation_count": current_generation_count + 1,
        "grounded": False,
        "useful": False,
        "needs_more_context": False,
    }


def hallucination_checker_node(
    state: GraphState,
) -> GraphState:
    check_result = check_hallucination(
        question=state["question"],
        answer=state["answer"],
        documents=state["documents"],
    )

    return {
        "grounded": check_result.grounded,
        "hallucination_reason": check_result.reason,
        "unsupported_claims": (
            check_result.unsupported_claims
        ),
    }


def answer_critic_node(state: GraphState) -> GraphState:
    critic_result = critique_answer(
        question=state["question"],
        answer=state["answer"],
        documents=state["documents"],
    )

    return {
        "useful": critic_result.useful,
        "needs_more_context": (
            critic_result.needs_more_context
        ),
        "critic_reason": critic_result.reason,
        "improvement_feedback": (
            critic_result.improvement_feedback
        ),
    }


def fallback_response_node(
    state: GraphState,
) -> GraphState:
    failure_reason = state.get(
        "hallucination_reason",
        "",
    )

    if state.get("grounded", False):
        failure_reason = state.get(
            "critic_reason",
            failure_reason,
        )

    return {
        "answer": (
            "I'm sorry, I don't know the answer "
            "to that question."
        ),
        "source_ids": [],
        "answer_source": "fallback",
        "failure_reason": failure_reason,
    }


def route_after_retrieval_decision(
    state: GraphState,
):
    if state["retrieval_needed"]:
        return "retrieve"

    return "direct_response"


def route_after_document_grading(
    state: GraphState,
):
    crag_route = state["crag_route"]

    if crag_route == "rewrite_query":
        rewrite_count = state.get(
            "rewrite_count",
            0,
        )

        max_retries = state.get(
            "max_retries",
            2,
        )

        if rewrite_count >= max_retries:
            return "search_web"

    return crag_route


def route_after_hallucination_check(
    state: GraphState,
):
    if state["grounded"]:
        return "critic"

    generation_count = state.get(
        "generation_count",
        0,
    )

    max_retries = state.get(
        "max_retries",
        2,
    )

    if generation_count >= max_retries:
        return "fallback"

    return "generate"


def route_after_answer_critic(
    state: GraphState,
):
    if state["useful"]:
        return "useful"

    needs_more_context = state.get(
        "needs_more_context",
        False,
    )

    if needs_more_context:
        answer_source = state.get(
            "answer_source",
            "documents",
        )

        # The graph already tried the web.
        if answer_source == "web":
            return "fallback"

        rewrite_count = state.get(
            "rewrite_count",
            0,
        )

        max_retries = state.get(
            "max_retries",
            2,
        )

        if rewrite_count >= max_retries:
            return "search_web"

        return "rewrite_query"

    generation_count = state.get(
        "generation_count",
        0,
    )

    max_retries = state.get(
        "max_retries",
        2,
    )

    if generation_count >= max_retries:
        return "fallback"

    return "generate"