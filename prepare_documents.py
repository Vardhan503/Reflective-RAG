import json
import os

from datasets import load_dataset


dataset_split = "validation"
number_of_examples = 20


if number_of_examples is None:
    selected_split = dataset_split
else:
    selected_split = dataset_split + "[:" + str(number_of_examples) + "]"


dataset = load_dataset(
    "hotpotqa/hotpot_qa",
    "distractor",
    split=selected_split,
)


corpus_documents = []
evaluation_questions = []
seen_titles = set()
duplicate_count = 0


for example in dataset:
    question_id = example["id"]

    supporting_titles = []

    for title in example["supporting_facts"]["title"]:
        if title not in supporting_titles:
            supporting_titles.append(title)

    evaluation_question = {
        "id": question_id,
        "question": example["question"],
        "reference_answer": example["answer"],
        "supporting_titles": supporting_titles,
        "question_type": example["type"],
        "level": example["level"],
    }

    evaluation_questions.append(evaluation_question)

    titles = example["context"]["title"]
    sentence_groups = example["context"]["sentences"]

    for document_number in range(len(titles)):
        title = titles[document_number]

        if title in seen_titles:
            duplicate_count = duplicate_count + 1
            continue

        seen_titles.add(title)

        sentences = sentence_groups[document_number]
        document_text = ""

        for sentence in sentences:
            document_text = document_text + sentence + " "

        document_id = "document_" + str(len(corpus_documents) + 1)

        document = {
            "id": document_id,
            "title": title,
            "text": document_text.strip(),
            "source": "hotpotqa",
        }

        corpus_documents.append(document)


os.makedirs("data", exist_ok=True)


with open("data/corpus.json", "w", encoding="utf-8") as file:
    json.dump(
        corpus_documents,
        file,
        indent=2,
        ensure_ascii=False,
    )


with open("data/evaluation_questions.json", "w", encoding="utf-8") as file:
    json.dump(
        evaluation_questions,
        file,
        indent=2,
        ensure_ascii=False,
    )


print("Dataset selection:", selected_split)
print("Corpus documents:", len(corpus_documents))
print("Evaluation questions:", len(evaluation_questions))
print("Duplicate documents skipped:", duplicate_count)
print("Corpus saved to: data/corpus.json")
print("Evaluation questions saved to: data/evaluation_questions.json")
