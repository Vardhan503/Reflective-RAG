import json
import os
import sys
import types

from dotenv import load_dotenv
from openai import AsyncOpenAI

# RAGAS 0.4.3 imports Vertex AI from langchain-community paths that were
# removed in langchain-community 0.4.2. Stub them before importing ragas so
# OpenAI evaluation can run without Google Vertex packages.
import langchain_community.chat_models
from langchain_community import llms as community_llms

if "langchain_community.chat_models.vertexai" not in sys.modules:
    vertexai_chat_module = types.ModuleType(
        "langchain_community.chat_models.vertexai"
    )

    class ChatVertexAI:
        pass

    vertexai_chat_module.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = vertexai_chat_module
    langchain_community.chat_models.vertexai = vertexai_chat_module

if not hasattr(community_llms, "VertexAI"):
    class VertexAI:
        pass

    community_llms.VertexAI = VertexAI

from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy
from ragas.metrics.collections import ContextPrecision
from ragas.metrics.collections import ContextRecall
from ragas.metrics.collections import Faithfulness
from ragas.metrics.collections import FactualCorrectness


load_dotenv()


INPUT_FILE = "data/evaluation_results.json"
OUTPUT_FILE = "data/ragas_evaluation_results.json"


# ---------------------------------------------------------
# Check the OpenAI API key
# ---------------------------------------------------------

openai_api_key = os.getenv("OPENAI_API_KEY")

if openai_api_key is None or openai_api_key.strip() == "":
    raise ValueError(
        "OPENAI_API_KEY was not found in the .env file."
    )


# ---------------------------------------------------------
# Create the RAGAS evaluator model
# ---------------------------------------------------------

openai_client = AsyncOpenAI(
    api_key=openai_api_key,
)

evaluator_llm = llm_factory(
    "gpt-4o-mini",
    client=openai_client,
)

evaluator_embeddings = embedding_factory(
    "openai",
    model="text-embedding-3-small",
    client=openai_client,
)


# ---------------------------------------------------------
# Create the RAGAS metrics
# ---------------------------------------------------------

faithfulness_metric = Faithfulness(
    llm=evaluator_llm,
)

answer_relevancy_metric = AnswerRelevancy(
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
)

context_precision_metric = ContextPrecision(
    llm=evaluator_llm,
)

context_recall_metric = ContextRecall(
    llm=evaluator_llm,
)

factual_correctness_metric = FactualCorrectness(
    llm=evaluator_llm,
    mode="f1",
)


# ---------------------------------------------------------
# Evaluate one RAG result
# ---------------------------------------------------------

def evaluate_result_with_ragas(evaluation_result):
    question = evaluation_result["user_input"]
    answer = evaluation_result["response"]
    reference_answer = evaluation_result["reference"]
    retrieved_contexts = evaluation_result[
        "retrieved_contexts"
    ]

    ragas_scores = {}

    print("\nQuestion:")
    print(question)

    # -----------------------------------------------------
    # Faithfulness
    # -----------------------------------------------------

    if len(retrieved_contexts) > 0:
        try:
            result = faithfulness_metric.score(
                user_input=question,
                response=answer,
                retrieved_contexts=retrieved_contexts,
            )

            ragas_scores["faithfulness"] = float(
                result.value
            )

        except Exception as error:
            print("Faithfulness failed:", error)
            ragas_scores["faithfulness"] = None

    else:
        ragas_scores["faithfulness"] = None

    # -----------------------------------------------------
    # Answer relevancy
    # -----------------------------------------------------

    try:
        result = answer_relevancy_metric.score(
            user_input=question,
            response=answer,
        )

        ragas_scores["answer_relevancy"] = float(
            result.value
        )

    except Exception as error:
        print("Answer relevancy failed:", error)
        ragas_scores["answer_relevancy"] = None

    # -----------------------------------------------------
    # Context precision
    # -----------------------------------------------------

    if len(retrieved_contexts) > 0:
        try:
            result = context_precision_metric.score(
                user_input=question,
                reference=reference_answer,
                retrieved_contexts=retrieved_contexts,
            )

            ragas_scores["context_precision"] = float(
                result.value
            )

        except Exception as error:
            print("Context precision failed:", error)
            ragas_scores["context_precision"] = None

    else:
        ragas_scores["context_precision"] = None

    # -----------------------------------------------------
    # Context recall
    # -----------------------------------------------------

    if len(retrieved_contexts) > 0:
        try:
            result = context_recall_metric.score(
                user_input=question,
                reference=reference_answer,
                retrieved_contexts=retrieved_contexts,
            )

            ragas_scores["context_recall"] = float(
                result.value
            )

        except Exception as error:
            print("Context recall failed:", error)
            ragas_scores["context_recall"] = None

    else:
        ragas_scores["context_recall"] = None

    # -----------------------------------------------------
    # Factual correctness
    # -----------------------------------------------------

    try:
        result = factual_correctness_metric.score(
            response=answer,
            reference=reference_answer,
        )

        ragas_scores["factual_correctness"] = float(
            result.value
        )

    except Exception as error:
        print("Factual correctness failed:", error)
        ragas_scores["factual_correctness"] = None

    print("Faithfulness:", ragas_scores["faithfulness"])
    print(
        "Answer relevancy:",
        ragas_scores["answer_relevancy"],
    )
    print(
        "Context precision:",
        ragas_scores["context_precision"],
    )
    print(
        "Context recall:",
        ragas_scores["context_recall"],
    )
    print(
        "Factual correctness:",
        ragas_scores["factual_correctness"],
    )

    updated_result = evaluation_result.copy()
    updated_result["ragas_scores"] = ragas_scores

    return updated_result


# ---------------------------------------------------------
# Calculate the average for one RAGAS metric
# ---------------------------------------------------------

def calculate_average(results, metric_name):
    total = 0.0
    number_of_scores = 0

    for result in results:
        ragas_scores = result["ragas_scores"]
        score = ragas_scores.get(metric_name)

        if score is not None:
            total = total + score
            number_of_scores = number_of_scores + 1

    if number_of_scores == 0:
        return 0.0

    average = total / number_of_scores

    return average


# ---------------------------------------------------------
# Run the complete RAGAS evaluation
# ---------------------------------------------------------

def run_ragas_evaluation():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            "data/evaluation_results.json was not found. "
            "Run evaluation_runner.py first."
        )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        evaluation_results = json.load(file)

    ragas_results = []

    for result_number in range(len(evaluation_results)):
        print(
            "Evaluating result",
            result_number + 1,
            "of",
            len(evaluation_results),
        )

        evaluation_result = evaluation_results[
            result_number
        ]

        updated_result = evaluate_result_with_ragas(
            evaluation_result
        )

        ragas_results.append(updated_result)

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            ragas_results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    average_faithfulness = calculate_average(
        ragas_results,
        "faithfulness",
    )

    average_answer_relevancy = calculate_average(
        ragas_results,
        "answer_relevancy",
    )

    average_context_precision = calculate_average(
        ragas_results,
        "context_precision",
    )

    average_context_recall = calculate_average(
        ragas_results,
        "context_recall",
    )

    average_factual_correctness = calculate_average(
        ragas_results,
        "factual_correctness",
    )

    print("RAGAS evaluation summary")

    print(
        "Questions evaluated:",
        len(ragas_results),
    )

    print(
        "Average faithfulness:",
        round(average_faithfulness, 4),
    )

    print(
        "Average answer relevancy:",
        round(average_answer_relevancy, 4),
    )

    print(
        "Average context precision:",
        round(average_context_precision, 4),
    )

    print(
        "Average context recall:",
        round(average_context_recall, 4),
    )

    print(
        "Average factual correctness:",
        round(average_factual_correctness, 4),
    )

    print(
        "Detailed results saved to:",
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    run_ragas_evaluation()