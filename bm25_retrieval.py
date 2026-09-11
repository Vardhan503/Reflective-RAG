import json
import re
from rank_bm25 import BM25Okapi

def tokenize(text):
    lowercase_text = text.lower()
    tokens = re.findall(r"\b\w+\b", lowercase_text)

    return tokens


def get_result_score(result):
    return result["score"]


with open("data/train_documents.json", "r", encoding="utf-8") as file:
    documents = json.load(file)

tokenized_documents = []

for document in documents:
    searchable_text = document["title"] + " " + document["text"]
    document_tokens = tokenize(searchable_text)
    tokenized_documents.append(document_tokens)

bm25 = BM25Okapi(tokenized_documents)

question = input("Enter a question: ").strip()

if question == "":
    question = "Which magazine was started first, Arthur's Magazine or First for Women?"

question_tokens = tokenize(question)
scores = bm25.get_scores(question_tokens)

ranked_results = []

for document_number in range(len(documents)):
    result = {
        "document": documents[document_number],
        "score": float(scores[document_number]),
    }

    ranked_results.append(result)

ranked_results.sort(key=get_result_score, reverse=True)
top_results = ranked_results[:5]

print("\nQuestion:")
print(question)

print("\nTop BM25 documents:")

for result_number in range(len(top_results)):
    result = top_results[result_number]
    document = result["document"]

    print("\nResult", result_number + 1)
    print("Title:", document["title"])
    print("BM25 score:", round(result["score"], 4))
    print("Text:", document["text"][:300])
