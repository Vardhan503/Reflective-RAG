import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

embedding_function = DefaultEmbeddingFunction()
client = chromadb.PersistentClient(path="data/chroma_db")
collection = client.get_collection(name="hotpotqa_train", embedding_function=embedding_function)


question = input("Enter a question: ")
if question == "":
    question = "What is the capital of France?"


results = collection.query(
    query_texts=[question],
    n_results=5,
    include=["documents", "metadatas", "distances"],
)
retrieved_documents = results["documents"][0]
retrieved_metadatas = results["metadatas"][0]
retrieved_distances = results["distances"][0]


print("\nQuestion:")
print(question)

print("\nTop retrieved documents:")

for result in range(len(retrieved_documents)):
    document_text = retrieved_documents[result]
    metadata = retrieved_metadatas[result]
    distance = retrieved_distances[result]
    similarity = 1 - distance

    print("\nResult", result + 1)
    print("Title:", metadata["title"])
    print("Similarity score:", round(similarity, 4))
    print("Text:", document_text[:300])
