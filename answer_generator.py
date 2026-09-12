from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

class GenerateAnswer(BaseModel):
    answer: str = Field(description="The answer to the question")
    
    source_ids: list[str] = Field(description="The IDs of the documents used to generate the answer")
    
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

structured_generator = model.with_structured_output(GenerateAnswer)

def build_context(documents):
    context = ""
    for document in documents:
        context+= "\nSource ID:" + document["id"]
        context+= "\nTitle:" + document["title"]
        context+= "\nContent:" + document["text"]
        document_url = document.get("url", "")
        if document_url:
            context+= "\nURL:" + document_url
        context+= "\n--------------------------------"
    print(f"Context: {context}")
    return context

def generate_answer(question, documents):
    if len(documents) == 0:
        raise ValueError("No documents provided")
    context = build_context(documents)
    prompt = f"""
You are the grounded answer generator in a RAG system.

Answer the user's question using only the provided documents.

Rules:

- Do not use outside knowledge.
- Do not invent missing facts.
- When the documents do not contain enough information, clearly say that
  there is not enough supported information to answer.
- Keep the answer concise and directly address the question.
- Mention source IDs inside square brackets after the claims they support.
- Return only source IDs that appear in the provided documents.

Question:
{question}

Documents:
{context}
"""


    response = structured_generator.invoke(prompt)
    print(f"Response: {response}")
    return response

if __name__ == "__main__":
    question = "Which magazine was started first, Arthur's Magazine or First for Women?"

    documents = [
        {
            "id": "document_1",
            "title": "Arthur's Magazine",
            "text": "Arthur's Magazine was an American literary periodical published from 1844 to 1846.",
        },
        {
            "id": "document_2",
            "title": "First for Women",
            "text": "First for Women is a women's magazine that began publishing in 1989.",
        },
    ]

    generated_result = generate_answer(
        question=question,
        documents=documents,
    )

    print("Question:")
    print(question)


    print("\nSource IDs:")

    for source_id in generated_result.source_ids:
        print(source_id)
