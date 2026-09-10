from datasets import load_dataset


dataset = load_dataset(
    "hotpotqa/hotpot_qa",
    "distractor",
    split="validation[:20]",
)

first_example = dataset[0]

print("Question:")
print(first_example["question"])

print("\nCorrect answer:")
print(first_example["answer"])

print("\nQuestion type:")
print(first_example["type"])

print("\nDocuments provided for this question:")

document_titles = first_example["context"]["title"]
document_sentences = first_example["context"]["sentences"]

for document_number in range(len(document_titles)):
    title = document_titles[document_number]
    sentences = document_sentences[document_number]

    print("\nDocument", document_number + 1)
    print("Title:", title)
    print("Text:")

    for sentence in sentences:
        print(sentence)

print("\nSupporting facts:")
print(first_example["supporting_facts"])
