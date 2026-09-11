import json
import os
from datasets import load_dataset


dataset_split = "train[:20]"

dataset = load_dataset(
    "hotpotqa/hotpot_qa",
    "distractor",
    split=dataset_split,
)

documents = []
seen_titles = set()
duplicate_count = 0

for example in dataset:
    question_id = example["id"]
    titles = example["context"]["title"]
    sentence_groups = example["context"]["sentences"]

    for document_number in range(len(titles)):
        title = titles[document_number]
        sentences = sentence_groups[document_number]

        if title in seen_titles:
            duplicate_count = duplicate_count + 1
            continue

        seen_titles.add(title)

        document_text = ""

        for sentence in sentences:
            document_text = document_text + sentence + " "

        document = {
            "id": question_id + "_" + str(document_number),
            "title": title,
            "text": document_text.strip(),
            "question_id": question_id,
        }

        documents.append(document)

print("Unique documents created:", len(documents))
print("Duplicate documents skipped:", duplicate_count)

print("\nFirst document:")
print(documents[0])

os.makedirs("data", exist_ok=True)

output_file = "data/train_documents.json"

with open(output_file, "w", encoding="utf-8") as file:
    json.dump(documents, file, indent=2, ensure_ascii=False)

print("\nDataset split:", dataset_split)
print("Documents saved to:", output_file)
