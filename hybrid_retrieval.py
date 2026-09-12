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


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[str]:
    """
    ranked_lists: e.g. [[d5, d11, d1], [d11, d5, d7], [d5, d7, d1]]
                  (each inner list is one retriever's/query's ranked doc ids, best first)

    Returns: single fused ranking, best first.
    """

    scores = defaultdict(float)

    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] += 1.0 / (k + rank)

    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def rerank_documents(query, candidates, top_k=5):
    model = CrossEncoder("BAAI/bge-reranker-v2-m3")

    pairs = [(query, candidate["text"]) for candidate in candidates]

    scores = model.predict(pairs)

    scored = sorted(
        zip(scores, candidates),
        key=lambda item: item[0],
        reverse=True,
    )

    return [candidate for _, candidate in scored[:top_k]]


with open("data/train_documents.json", "r", encoding="utf-8") as file:
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

client = chromadb.PersistentClient(path="data/chroma_db")

collection = client.get_collection(
    name="hotpotqa_train",
    embedding_function=embedding_function,
)

question = input("Enter a question: ").strip()

if question == "":
    question = "Which magazine was started first, Arthur's Magazine or First for Women?"

retrieval_count = 10

dense_results = collection.query(
    query_texts=[question],
    n_results=retrieval_count,
)

dense_document_ids = dense_results["ids"][0]

question_tokens = tokenize(question)
bm25_scores = bm25.get_scores(question_tokens)

bm25_results = []

for document_number in range(len(documents)):
    bm25_result = {
        "document_id": documents[document_number]["id"],
        "score": float(bm25_scores[document_number]),
    }

    bm25_results.append(bm25_result)

bm25_results.sort(key=get_result_score, reverse=True)
top_bm25_results = bm25_results[:retrieval_count]

bm25_document_ids = []

for result in top_bm25_results:
    bm25_document_ids.append(result["document_id"])

ranked_lists = [dense_document_ids, bm25_document_ids]
fused_document_ids = reciprocal_rank_fusion(ranked_lists)

reranker_candidates = []
top_fused_document_ids = fused_document_ids[:10]

for document_id in top_fused_document_ids:
    document = documents_by_id[document_id]
    reranker_candidates.append(document)

reranked_documents = rerank_documents(
    query=question,
    candidates=reranker_candidates,
    top_k=5,
)

print("\nQuestion:")
print(question)

print("\nTop documents after cross-encoder reranking:")

for result_number in range(len(reranked_documents)):
    document = reranked_documents[result_number]
    document_id = document["id"]

    dense_rank = "Not in dense top 10"
    bm25_rank = "Not in BM25 top 10"
    rrf_rank = fused_document_ids.index(document_id) + 1

    if document_id in dense_document_ids:
        dense_rank = dense_document_ids.index(document_id) + 1

    if document_id in bm25_document_ids:
        bm25_rank = bm25_document_ids.index(document_id) + 1

    print("\nResult", result_number + 1)
    print("Title:", document["title"])
    print("RRF rank:", rrf_rank)
    print("Dense rank:", dense_rank)
    print("BM25 rank:", bm25_rank)
    print("Text:", document["text"][:300])
