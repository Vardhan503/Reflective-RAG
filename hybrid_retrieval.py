import json
import re
from collections import defaultdict

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


def tokenize(text):
    text = text.lower()
    tokens = re.findall(r"\b\w+\b", text)

    return tokens


def get_score(result):
    return result["score"]


def reciprocal_rank_fusion(ranked_lists, k=60):
    scores = defaultdict(float)

    for ranked_list in ranked_lists:

        for rank, document_id in enumerate(ranked_list, start=1):

            scores[document_id] += 1.0 / (k + rank)

    sorted_results = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    fused_document_ids = []

    for document_id, score in sorted_results:
        fused_document_ids.append(document_id)

    return fused_document_ids


def rerank_documents(query, candidates, model, top_k=5):
    pairs = []

    for candidate in candidates:

        document_text = candidate["title"] + " " + candidate["text"]

        pair = (query, document_text)

        pairs.append(pair)

    scores = model.predict(pairs)

    scored_documents = []

    for index in range(len(candidates)):

        result = {
            "score": float(scores[index]),
            "document": candidates[index],
        }

        scored_documents.append(result)

    scored_documents.sort(
        key=get_score,
        reverse=True,
    )

    reranked_documents = []

    top_results = scored_documents[:top_k]

    for result in top_results:

        document = result["document"].copy()

        document["reranker_score"] = result["score"]

        reranked_documents.append(document)

    return reranked_documents


# ---------------------------------------------------------
# Load the searchable HotpotQA corpus
# ---------------------------------------------------------

with open("data/corpus.json", "r", encoding="utf-8") as file:
    documents = json.load(file)


documents_by_id = {}
tokenized_documents = []


for document in documents:

    document_id = document["id"]

    documents_by_id[document_id] = document

    searchable_text = document["title"] + " " + document["text"]

    tokens = tokenize(searchable_text)

    tokenized_documents.append(tokens)


# ---------------------------------------------------------
# Create the BM25 sparse retriever
# ---------------------------------------------------------

bm25 = BM25Okapi(tokenized_documents)


# ---------------------------------------------------------
# Connect to the existing Chroma vector database
# ---------------------------------------------------------

embedding_function = DefaultEmbeddingFunction()

chroma_client = chromadb.PersistentClient(
    path="data/chroma_db",
)

collection = chroma_client.get_collection(
    name="hotpotqa_corpus",
    embedding_function=embedding_function,
)


# ---------------------------------------------------------
# Load the cross-encoder reranker
# ---------------------------------------------------------

reranker_model = CrossEncoder(
    "BAAI/bge-reranker-v2-m3"
)


# ---------------------------------------------------------
# Dense retrieval using ChromaDB
# ---------------------------------------------------------

def dense_retrieve(question, retrieval_count):
    number_of_results = retrieval_count

    if collection.count() < retrieval_count:
        number_of_results = collection.count()

    results = collection.query(
        query_texts=[question],
        n_results=number_of_results,
    )

    document_ids = results["ids"][0]

    return document_ids


# ---------------------------------------------------------
# Sparse retrieval using BM25
# ---------------------------------------------------------

def bm25_retrieve(question, retrieval_count):
    question_tokens = tokenize(question)

    scores = bm25.get_scores(question_tokens)

    scored_documents = []

    for index in range(len(documents)):

        result = {
            "document_id": documents[index]["id"],
            "score": float(scores[index]),
        }

        scored_documents.append(result)

    scored_documents.sort(
        key=get_score,
        reverse=True,
    )

    top_results = scored_documents[:retrieval_count]

    document_ids = []

    for result in top_results:
        document_ids.append(result["document_id"])

    return document_ids


# ---------------------------------------------------------
# Complete hybrid retrieval pipeline
# ---------------------------------------------------------

def hybrid_retrieve(question, retrieval_count=10, top_k=5):
    question = question.strip()

    if question == "":
        raise ValueError("Question cannot be empty.")

    dense_document_ids = dense_retrieve(
        question=question,
        retrieval_count=retrieval_count,
    )

    bm25_document_ids = bm25_retrieve(
        question=question,
        retrieval_count=retrieval_count,
    )

    ranked_lists = [
        dense_document_ids,
        bm25_document_ids,
    ]

    fused_document_ids = reciprocal_rank_fusion(
        ranked_lists=ranked_lists,
        k=60,
    )

    top_fused_document_ids = fused_document_ids[:retrieval_count]

    reranker_candidates = []

    for document_id in top_fused_document_ids:

        if document_id in documents_by_id:

            document = documents_by_id[document_id].copy()

            document["source"] = "hybrid_retrieval"

            reranker_candidates.append(document)

    reranked_documents = rerank_documents(
        query=question,
        candidates=reranker_candidates,
        model=reranker_model,
        top_k=top_k,
    )

    return reranked_documents


