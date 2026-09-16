import json
import os
import re
import string
from collections import Counter

from rag_graph import run_rag


def normalize_text(text):
    text = text.lower()
    text = text.translate(
        str.maketrans("", "", string.punctuation))
    words = text.split()
    normalized_words = []
    for word in words:
        if word not in ["a", "an", "the"]:
            normalized_words.append(word)

    normalized_text = " ".join(normalized_words)
    return normalized_text


def calculate_exact_match(generated_answer, reference_answer):
    generated_answer = normalize_text(generated_answer)
    reference_answer = normalize_text(reference_answer)

    if generated_answer == reference_answer:
        return 1.0

    return 0.0


def calculate_answer_f1(generated_answer, reference_answer):
    generated_tokens = normalize_text(generated_answer).split()
    reference_tokens = normalize_text(reference_answer).split()
    if len(generated_tokens) == 0 or len(reference_tokens) == 0:
        return 0.0

    generated_counts = Counter(generated_tokens)
    reference_counts = Counter(reference_tokens)

    common_tokens = generated_counts & reference_counts
    number_of_common_tokens = sum(common_tokens.values())

    if number_of_common_tokens == 0:
        return 0.0

    precision = number_of_common_tokens / len(generated_tokens)
    recall = number_of_common_tokens / len(reference_tokens)

    f1_score = 2 * precision * recall
    f1_score = f1_score / (precision + recall)

    return f1_score


def calculate_title_recall(retrieved_titles, supporting_titles):
    if len(supporting_titles) == 0:
        return 0.0

    normalized_retrieved_titles = []

    for title in retrieved_titles:
        normalized_title = normalize_text(title)
        normalized_retrieved_titles.append(normalized_title)

    number_of_matches = 0

    for supporting_title in supporting_titles:
        normalized_supporting_title = normalize_text(
            supporting_title
        )

        if normalized_supporting_title in normalized_retrieved_titles:
            number_of_matches = number_of_matches + 1

    title_recall = number_of_matches / len(supporting_titles)

    return title_recall


def get_document_information(documents):
    retrieved_contexts = []
    retrieved_titles = []

    for document in documents:
        retrieved_contexts.append(document["text"])
        retrieved_titles.append(document["title"])

    return retrieved_contexts, retrieved_titles


def evaluate_question(evaluation_example):
    question = evaluation_example["question"]
    reference_answer = evaluation_example["reference_answer"]
    supporting_titles = evaluation_example["supporting_titles"]

    print("\nEvaluating:")
    print(question)

    rag_result = run_rag(
        question=question,
        max_retries=2,
    )

    generated_answer = rag_result.get("answer", "")
    documents = rag_result.get("documents", [])

    retrieved_contexts, retrieved_titles = get_document_information(
        documents
    )

    exact_match = calculate_exact_match(
        generated_answer,
        reference_answer,
    )

    answer_f1 = calculate_answer_f1(
        generated_answer,
        reference_answer,
    )

    title_recall = calculate_title_recall(
        retrieved_titles,
        supporting_titles,
    )

    evaluation_result = {
        "id": evaluation_example["id"],

        # RAGAS-compatible fields
        "user_input": question,
        "response": generated_answer,
        "reference": reference_answer,
        "retrieved_contexts": retrieved_contexts,

        # HotpotQA retrieval information
        "supporting_titles": supporting_titles,
        "retrieved_titles": retrieved_titles,

        # Basic evaluation metrics
        "exact_match": exact_match,
        "answer_f1": answer_f1,
        "supporting_title_recall": title_recall,

        # CRAG and Self-RAG information
        "answer_source": rag_result.get(
            "answer_source",
            "unknown",
        ),
        "retrieval_needed": rag_result.get(
            "retrieval_needed",
            False,
        ),
        "router_reason": rag_result.get(
            "router_reason",
            "",
        ),
        "crag_route": rag_result.get(
            "crag_route",
            "",
        ),
        "grounded": rag_result.get(
            "grounded",
            False,
        ),
        "hallucination_reason": rag_result.get(
            "hallucination_reason",
            "",
        ),
        "useful": rag_result.get(
            "useful",
            False,
        ),
        "critic_reason": rag_result.get(
            "critic_reason",
            "",
        ),
        "rewrite_count": rag_result.get(
            "rewrite_count",
            0,
        ),
        "generation_count": rag_result.get(
            "generation_count",
            0,
        ),
        "elapsed_seconds": rag_result.get(
            "elapsed_seconds",
            0.0,
        ),
    }

    print("Generated answer:", generated_answer)
    print("Reference answer:", reference_answer)
    print("Exact match:", exact_match)
    print("Answer F1:", round(answer_f1, 4))
    print("Supporting title recall:", round(title_recall, 4))

    return evaluation_result


def calculate_average(results, metric_name):
    total = 0.0

    for result in results:
        total = total + result[metric_name]

    if len(results) == 0:
        return 0.0

    average = total / len(results)

    return average


def run_evaluation(number_of_questions=5):
    with open(
        "data/evaluation_questions.json",
        "r",
        encoding="utf-8",
    ) as file:
        evaluation_questions = json.load(file)

    selected_questions = evaluation_questions[
        :number_of_questions
    ]

    evaluation_results = []

    for example in selected_questions:
        try:
            result = evaluate_question(example)
            evaluation_results.append(result)

        except Exception as error:
            print("\nEvaluation failed:")
            print(example["question"])
            print("Error:", error)

    os.makedirs("data", exist_ok=True)

    with open(
        "data/evaluation_results.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            evaluation_results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    average_exact_match = calculate_average(
        evaluation_results,
        "exact_match",
    )

    average_answer_f1 = calculate_average(
        evaluation_results,
        "answer_f1",
    )

    average_title_recall = calculate_average(
        evaluation_results,
        "supporting_title_recall",
    )

    print("Evaluation summary")
    print("Questions evaluated:", len(evaluation_results))
    print("Average exact match:", round(average_exact_match, 4))
    print("Average answer F1:", round(average_answer_f1, 4))
    print(
        "Average supporting title recall:",
        round(average_title_recall, 4),
    )
    print(
        "Results saved to:",
        "data/evaluation_results.json",
    )


if __name__ == "__main__":
    run_evaluation(number_of_questions=5)