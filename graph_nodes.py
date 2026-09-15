from answer_critic import critique_answer
from answer_generator import generate_answer
from crag_document_grader import grade_documents, choose_crag_route
from hallucination_checker import check_hallucination
from graph_state import GraphState
from hybrid_retrieval import hybrid_retrieve
from models import generator_model
from query_rewriter import rewrite_query
from retrieval_router import decide_if_retrieval_is_needed
from web_search import search_web


def retrieval_router_node(state: GraphState) -> GraphState:
    decision = decide_if_retrieval_is_needed(state["question"])
    return {**state, "retrieval_needed": decision.retrieve, "router_reason": decision.reason}

def direct_response_node(state: GraphState) -> GraphState:
    question = state["question"]
    prompt = f"""
    Respond directly to the user's question: {question}
    This request does not require external documents.
    Do not claim that you searched a knowledge base or the web"""
    response = generator_model.invoke(prompt)
    return {
        **state,
        "answer": response.content,
        "source_ids": [],
        "grounded": True,
        "useful": True,
        "answer_source": "direct",
    }

def retrieve_documents_node(state: GraphState) -> GraphState:
    retrieval_query = state.get("retrieval_query", state["question"])
    documents = hybrid_retrieve(retrieval_query)
    return {**state, "retrieval_query": retrieval_query, "documents": documents}

def grade_documents_node(state: GraphState) -> GraphState:
    question = state["question"]
    documents = state["documents"]
    graded_documents = grade_documents(question, documents)
    crag_route = choose_crag_route(graded_documents)
    result = {"graded_documents": graded_documents, "crag_route":crag_route}
    if crag_route == "generate":
        correct_documents = []
        for document in graded_documents:
            if document["grade"] == "correct":
                correct_documents.append(document)
        result["documents"] = correct_documents
    return result

def rewrite_query_node(state: GraphState) -> GraphState:
    question = state["question"]
    graded_documents = state["graded_documents"]
    ambiguous_documents = []
    for document in graded_documents:
        if document["grade"] == "ambiguous":
            ambiguous_documents.append(document)
    rewrite_result = rewrite_query(question, ambiguous_documents)
    current_rewrite_count = state.get("rewrite_count", 0)
    return {
        **state,
        "retrieval_query": rewrite_result.rewritten_query,
        "rewrite_count": current_rewrite_count + 1,
    }
    
def web_search_node(state: GraphState) -> GraphState:
    search_query = state.get("retrieval_query", state["question"])
    web_documents = search_web(search_query, max_results=5)
    return {
        **state,
        "documents": web_documents,
        "graded_documents": [],
        "crag_route": "generate",
        "answer_source": "web",
    }

def generate_answer_node(state: GraphState):
    question = state["question"]
    documents = state["documents"]
    generation_question = question
    improvement_feedback = state.get("improvement_feedback", "")

    if improvement_feedback != "":
        generation_question = generation_question + "\n\n"
        generation_question = generation_question + "Regeneration instructions: "
        generation_question = generation_question + improvement_feedback
    unsupported_claims = state.get("unsupported_claims", [])
    if len(unsupported_claims) > 0:
        generation_question = generation_question + "\n\n"
        generation_question = generation_question + "Do not repeat these unsupported claims:"
        for claim in unsupported_claims:
            generation_question = generation_question + "\n- " + claim
    generated_result = generate_answer(question=generation_question, documents=documents)
    current_generation_count = state.get("generation_count", 0)
    return {
        "answer": generated_result.answer,
        "source_ids": generated_result.source_ids,
        "generation_count": current_generation_count + 1,
        "grounded": False,
        "useful": False,
    }

def hallucination_checker_node(state: GraphState) -> GraphState:
    check_result=check_hallucination(state["question"], state["answer"], state["documents"])
    return {**state, "grounded": check_result.grounded, "hallucination_reason": check_result.reason, "unsupported_claims": check_result.unsupported_claims}

def answer_critic_node(state: GraphState) -> GraphState:
    critic_result = critique_answer(state["question"], state["answer"], state["documents"])
    return {**state, "useful": critic_result.useful, "critic_reason": critic_result.reason, "improvement_feedback": critic_result.improvement_feedback}

def fallback_response_node(state:GraphState) -> GraphState:
    return {
        **state,
        "answer": ("I'm sorry, I don't know the answer to that question."),
        "source_ids": [],
        "answer_source": "fallback",
    }

def route_after_retrival_decision(state:GraphState) -> GraphState:
    if state["retrieval_needed"]:
        return "retrieve"
    return "direct_response"

def route_after_document_grading(state:GraphState) -> GraphState:
    crag_route = state["crag_route"]
    if crag_route == "rewrite_query":
        rewrite_count = state.get("rewrite_count", 0)
        max_retries = state.get("max_retries", 2)
        if rewrite_count >= max_retries:
            return "search_web"
    return crag_route

def router_after_hallucination_check(state:GraphState) -> GraphState:
    if state["grounded"]:
        return "critic"
    generation_count = state.get("generation_count", 0)
    max_retries = state.get("max_retries", 2)
    if generation_count >= max_retries:
        return "fallback"
    return "generate"

def router_after_answer_critic(state:GraphState) -> GraphState:
    if state["useful"]:
        return "useful"
    generation_count = state.get("generation_count", 0)
    max_retries = state.get("max_retries", 2)
    if generation_count >= max_retries:
        return "fallback"
    return "generate"
    
    