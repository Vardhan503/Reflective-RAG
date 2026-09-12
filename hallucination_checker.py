from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


load_dotenv()


class HallucinationResult(BaseModel):
    grounded: bool = Field(
        description="True when every factual claim is supported by the documents."
    )

    reason: str = Field(
        description="A short explanation of the grounding decision."
    )

    unsupported_claims: list[str] = Field(
        description="Factual claims from the answer that the documents do not support."
    )


model = ChatOpenAI(model="gpt-5-mini")

structured_checker = model.with_structured_output(HallucinationResult)


def build_context(documents):
    context = ""

    for document in documents:
        context = context + "\nSource ID: " + document["id"]
        context = context + "\nTitle: " + document["title"]
        context = context + "\nText: " + document["text"]
        context = context + "\n"

    return context


def check_hallucination(question, answer, documents):
    if len(documents) == 0:
        raise ValueError("At least one document is required for grounding verification.")

    context = build_context(documents)

    prompt = f"""
You are the hallucination checker in a RAG system.

Determine whether every factual claim in the generated answer is supported
by the provided documents.

Rules:

- Break the answer into individual factual claims.
- A claim is supported only when the document text provides evidence for it.
- A source ID written in the answer is not evidence by itself.
- If even one factual claim is unsupported, grounded must be false.
- Copy every unsupported claim into unsupported_claims.
- Do not use outside knowledge when checking the answer.
- Ignore non-factual conversational phrases.

Question:
{question}

Generated answer:
{answer}

Documents:
{context}
"""

    result = structured_checker.invoke(prompt)

    return result


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

    answer = """
Arthur's Magazine started first in 1844, while First for Women began in 1989.
Both magazines were founded in Philadelphia.
"""

    check_result = check_hallucination(
        question=question,
        answer=answer,
        documents=documents,
    )

    print("Grounded:", check_result.grounded)
    print("Reason:", check_result.reason)

    print("\nUnsupported claims:")

    if len(check_result.unsupported_claims) == 0:
        print("None")
    else:
        for claim in check_result.unsupported_claims:
            print("-", claim)
