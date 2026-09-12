import json
import re
from collections import defaultdict

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


def tokenize(text):
    lowercase_text = text.lower()
    tokens = re.findall(r"\b\w+\b", lowercase_text)

    return tokens


def get_result_score(result):
    return result["score"]


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[str]:

    scores = defaultdict(float)

    for ranked in ranked_lists:
        for rank, document_id in enumerate(ranked, start=1):
            scores[document_id] += 1.0 / (k + rank)

    sorted_scores = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    fused_document_ids = []

    for document_id, score in sorted_scores:
        fused_document_ids.append(document_id)

    return fused_document_ids


def rerank_documents(query, candidates, model, top_k=5):
    pairs = []

    for candidate in candidates:
        candidate_text = candidate["title"] + " " + candidate["text"]

        pair = (query, candidate_text)

        pairs.append(pair)

    scores = model.predict(pairs)

    scored_documents = []

    for result_number in range(len(candidates)):
        result = {
            "score": float(scores[result_number]),
            "document": candidates[result_number],
        }

        scored_documents.append(result)

    scored_documents.sort(
        key=get_result_score,
        reverse=True,
    )

    top_results = scored_documents[:top_k]

    reranked_documents = []

    for result in top_results:
        document = result["document"]
        reranked_documents.append(document)

    return reranked_documents


with open(
    "data/train_documents.json",
    "r",
    encoding="utf-8",
) as file:
    documents = json.load(file)


documents_by_id = {}
tokenized_documents = []

for document in documents:
    document_id = document["id"]

    documents_by_id[document_id] = document

    searchable_text = document["title"] + " " + document["text"]

    document_tokens = tokenize(searchable_text)

    tokenized_documents.append(document_tokens)


bm25 = BM25Okapi(tokenized_documents)


embedding_function = DefaultEmbeddingFunction()

client = chromadb.PersistentClient(
    path="data/chroma_db",
)

collection = client.get_collection(
    name="hotpotqa_train",
    embedding_function=embedding_function,
)


reranker_model = CrossEncoder(
    "BAAI/bge-reranker-v2-m3"
)


def hybrid_retrieve(
    question,
    retrieval_count=10,
    top_k=5,
):
    if question.strip() == "":
        raise ValueError("Question cannot be empty.")

    # Step 1: Dense retrieval from ChromaDB

    dense_results = collection.query(
        query_texts=[question],
        n_results=retrieval_count,
    )

    dense_document_ids = dense_results["ids"][0]

    # Step 2: Sparse retrieval using BM25

    question_tokens = tokenize(question)

    bm25_scores = bm25.get_scores(question_tokens)

    bm25_results = []

    for document_number in range(len(documents)):
        bm25_result = {
            "document_id": documents[document_number]["id"],
            "score": float(bm25_scores[document_number]),
        }

        bm25_results.append(bm25_result)

    bm25_results.sort(
        key=get_result_score,
        reverse=True,
    )

    top_bm25_results = bm25_results[:retrieval_count]

    bm25_document_ids = []

    for result in top_bm25_results:
        bm25_document_ids.append(
            result["document_id"]
        )

    # Step 3: Combine rankings using RRF

    ranked_lists = [
        dense_document_ids,
        bm25_document_ids,
    ]

    fused_document_ids = reciprocal_rank_fusion(
        ranked_lists
    )

    top_fused_document_ids = fused_document_ids[
        :retrieval_count
    ]

    # Step 4: Get complete documents

    reranker_candidates = []

    for document_id in top_fused_document_ids:
        document = documents_by_id[document_id].copy()

        document["source"] = "vector_database"

        reranker_candidates.append(document)

    # Step 5: Cross-encoder reranking

    reranked_documents = rerank_documents(
        query=question,
        candidates=reranker_candidates,
        model=reranker_model,
        top_k=top_k,
    )

    return reranked_documents


if __name__ == "__main__":
    question = input("Enter a question: ").strip()

    if question == "":
        question = (
            "Which magazine was started first, "
            "Arthur's Magazine or First for Women?"
        )

    retrieved_documents = hybrid_retrieve(
        question=question,
        retrieval_count=10,
        top_k=5,
    )

    print("\nQuestion:")
    print(question)

    print(
        "\nTop documents after hybrid retrieval "
        "and reranking:"
    )

    for result_number in range(
        len(retrieved_documents)
    ):
        document = retrieved_documents[result_number]

        print("\nResult", result_number + 1)
        print("Title:", document["title"])
        print("Document ID:", document["id"])
        print("Source:", document["source"])
        print("Text:", document["text"][:300])