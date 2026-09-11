import json

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

with open("data/train_documents.json", "r", encoding="utf-8") as file:
    documents = json.load(file)
    
document_ids = []
document_texts = []
document_metadatas = []

for document in documents:
    document_ids.append(document["id"])
    text_for_embedding = document["title"] + " " + document["text"]
    document_texts.append(text_for_embedding)
    document_metadatas.append({"title": document["title"], "question_id": document["question_id"]})

embedding_function = DefaultEmbeddingFunction()
client = chromadb.PersistentClient(path="data/chroma_db")
collection = client.get_or_create_collection(
    name="hotpotqa_train",
    embedding_function=embedding_function,
    metadata={"hnsw:space": "cosine"},
)

collection.upsert(
    documents=document_texts,
    ids=document_ids,
    metadatas=document_metadatas,
)

print(f"Vector database built with {len(documents)} documents")
print(f"Collection: {collection.name}")
print("Documents stored in ChromaDB:", collection.count())
