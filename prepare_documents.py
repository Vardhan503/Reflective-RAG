from datasets import load_dataset


dataset = load_dataset(
    "hotpotqa/hotpot_qa",
    "distractor",
    split="validation[:20]",
)

documents = []

for example in dataset:
    question_id = example["id"]
    
    titles = example["context"]["title"]
    sentence_groups = example["context"]["sentences"]

    for document_number in range(2):
        title = titles[document_number]
        sentences = sentence_groups[document_number]


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

print("Total documents created:", len(documents))

print("\nFirst document:")
print(documents[0])
