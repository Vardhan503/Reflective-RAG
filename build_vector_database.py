import json

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction


with open("data/corpus.json", "r", encoding="utf-8") as file:
    documents = json.load(file)


document_ids = []
document_texts = []
document_metadatas = []


for document in documents:
    document_ids.append(document["id"])

    text_for_embedding = document["title"] + " " + document["text"]
    document_texts.append(text_for_embedding)

    metadata = {
        "title": document["title"],
        "source": document["source"],
    }

    document_metadatas.append(metadata)


embedding_function = DefaultEmbeddingFunction()

client = chromadb.PersistentClient(
    path="data/chroma_db",
)

collection = client.get_or_create_collection(
    name="hotpotqa_corpus",
    embedding_function=embedding_function,
    metadata={"hnsw:space": "cosine"},
)


batch_size = 1000
batch_start = 0


while batch_start < len(documents):
    batch_end = batch_start + batch_size

    collection.upsert(
        ids=document_ids[batch_start:batch_end],
        documents=document_texts[batch_start:batch_end],
        metadatas=document_metadatas[batch_start:batch_end],
    )

    print(
        "Stored documents:",
        min(batch_end, len(documents)),
        "of",
        len(documents),
    )

    batch_start = batch_end


print("Collection:", collection.name)
print("Documents in ChromaDB:", collection.count())
print("ChromaDB folder: data/chroma_db")
