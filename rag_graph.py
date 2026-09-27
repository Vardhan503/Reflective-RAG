import json
import time

from langgraph.graph import END, START, StateGraph

from graph_nodes import (
    answer_critic_node,
    direct_response_node,
    fallback_response_node,
    generate_answer_node,
    grade_documents_node,
    hallucination_checker_node,
    retrieve_documents_node,
    retrieval_router_node,
    rewrite_query_node,
    route_after_answer_critic,
    route_after_document_grading,
    route_after_hallucination_check,
    route_after_retrieval_decision,
    web_search_node,
)
from graph_state import GraphState


graph_builder = StateGraph(GraphState)


graph_builder.add_node(
    "retrieval_router",
    retrieval_router_node,
)

graph_builder.add_node(
    "direct_response",
    direct_response_node,
)

graph_builder.add_node(
    "retrieve_documents",
    retrieve_documents_node,
)

graph_builder.add_node(
    "grade_documents",
    grade_documents_node,
)

graph_builder.add_node(
    "rewrite_query",
    rewrite_query_node,
)

graph_builder.add_node(
    "web_search",
    web_search_node,
)

graph_builder.add_node(
    "generate_answer",
    generate_answer_node,
)

graph_builder.add_node(
    "hallucination_checker",
    hallucination_checker_node,
)

graph_builder.add_node(
    "answer_critic",
    answer_critic_node,
)

graph_builder.add_node(
    "fallback_response",
    fallback_response_node,
)


graph_builder.add_edge(
    START,
    "retrieval_router",
)


graph_builder.add_conditional_edges(
    "retrieval_router",
    route_after_retrieval_decision,
    {
        "retrieve": "retrieve_documents",
        "direct_response": "direct_response",
    },
)


graph_builder.add_edge(
    "direct_response",
    END,
)


graph_builder.add_edge(
    "retrieve_documents",
    "grade_documents",
)


graph_builder.add_conditional_edges(
    "grade_documents",
    route_after_document_grading,
    {
        "generate": "generate_answer",
        "rewrite_query": "rewrite_query",
        "search_web": "web_search",
    },
)


graph_builder.add_edge(
    "rewrite_query",
    "retrieve_documents",
)


graph_builder.add_edge(
    "web_search",
    "generate_answer",
)


graph_builder.add_edge(
    "generate_answer",
    "hallucination_checker",
)


graph_builder.add_conditional_edges(
    "hallucination_checker",
    route_after_hallucination_check,
    {
        "critic": "answer_critic",
        "generate": "generate_answer",
        "fallback": "fallback_response",
    },
)


graph_builder.add_conditional_edges(
    "answer_critic",
    route_after_answer_critic,
    {
        "useful": END,
        "generate": "generate_answer",
        "rewrite_query": "rewrite_query",
        "search_web": "web_search",
        "fallback": "fallback_response",
    },
)


graph_builder.add_edge(
    "fallback_response",
    END,
)


rag_graph = graph_builder.compile()


def run_rag(
    question,
    max_retries=2,
    run_label="development",
):
    if question.strip() == "":
        raise ValueError("Question cannot be empty.")

    initial_state: GraphState = {
        "question": question,
        "retrieval_query": question,

        "retrieval_needed": False,
        "router_reason": "",

        "documents": [],
        "accumulated_documents": [],
        "graded_documents": [],

        "crag_route": "",
        "context_status": "",
        "context_reason": "",
        "missing_information": "",

        "answer": "",
        "last_generated_answer": "",
        "source_ids": [],
        "answer_source": "documents",

        "grounded": False,
        "hallucination_reason": "",
        "unsupported_claims": [],

        "useful": False,
        "needs_more_context": False,
        "critic_reason": "",
        "improvement_feedback": "",

        "rewrite_count": 0,
        "generation_count": 0,
        "max_retries": max_retries,

        "web_search_used": False,
        "failure_reason": "",
    }

    graph_config = {
        "recursion_limit": 50,
        "run_name": "reflective-rag",
        "tags": [
            "crag",
            "self-rag",
            run_label,
        ],
        "metadata": {
            "run_label": run_label,
            "max_retries": max_retries,
        },
    }

    start_time = time.perf_counter()

    final_state = rag_graph.invoke(
        initial_state,
        config=graph_config,
    )

    elapsed_seconds = (
        time.perf_counter() - start_time
    )

    final_state["elapsed_seconds"] = elapsed_seconds

    return final_state


def print_rag_result(result, label=""):
    if label != "":
        print("\n" + "=" * 60)
        print(label)
        print("=" * 60)

    answer_source = result.get(
        "answer_source",
        "documents",
    )

    elapsed_seconds = result.get(
        "elapsed_seconds",
        0.0,
    )

    print("\nAnswer source:", answer_source)
    print("Context status:", result.get("context_status", ""))
    print("Rewrite count:", result.get("rewrite_count", 0))
    print("Generation count:", result.get("generation_count", 0))
    print("Web search used:", result.get("web_search_used", False))
    print(f"Time: {elapsed_seconds:.2f}s")

    print("\nAnswer:")
    print(result["answer"])

    source_ids = result.get("source_ids", [])

    if len(source_ids) > 0:
        print("\nSource IDs:")

        for source_id in source_ids:
            print("-", source_id)

    failure_reason = result.get(
        "failure_reason",
        "",
    )

    if failure_reason != "":
        print("\nFailure reason:")
        print(failure_reason)


def run_timing_comparison(max_retries=2):
    with open(
        "data/evaluation_questions.json",
        "r",
        encoding="utf-8",
    ) as file:
        evaluation_questions = json.load(file)

    in_corpus_question = evaluation_questions[0][
        "question"
    ]

    out_of_corpus_question = (
        "What was the closing price of NVIDIA stock "
        "on the most recent trading day?"
    )

    in_corpus_result = run_rag(
        question=in_corpus_question,
        max_retries=max_retries,
        run_label="timing-in-corpus",
    )

    print_rag_result(
        in_corpus_result,
        label="In-corpus question",
    )

    out_of_corpus_result = run_rag(
        question=out_of_corpus_question,
        max_retries=max_retries,
        run_label="timing-out-of-corpus",
    )

    print_rag_result(
        out_of_corpus_result,
        label="Out-of-corpus question",
    )


if __name__ == "__main__":
    import sys

    if (
        len(sys.argv) > 1
        and sys.argv[1] == "--test-timing"
    ):
        run_timing_comparison()

    else:
        user_question = input(
            "Enter a question: "
        ).strip()

        if user_question == "":
            print("Please enter a question.")

        else:
            result = run_rag(
                question=user_question,
                run_label="manual-test",
            )

            print_rag_result(result)