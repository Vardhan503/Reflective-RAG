from typing import Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


load_dotenv()


class DocumentGrade(BaseModel):
    grade: Literal["correct", "ambiguous", "incorrect"] = Field(
        description="How useful the document is for answering the question."
    )

    reason: str = Field(
        description="A short explanation for the selected grade."
    )


model = ChatOpenAI(model="gpt-5-mini")

structured_grader = model.with_structured_output(DocumentGrade)


def grade_document(question, document_text):
    prompt = f"""
You are a document relevance grader for a Corrective RAG system.

Compare the document with the user's question.

Use one of these grades:

- correct: The document contains information that directly helps answer the question.
- ambiguous: The document is related, but it does not provide enough information to answer confidently.
- incorrect: The document is unrelated or not useful for answering the question.

The word incorrect means irrelevant for this question. It does not necessarily mean
that the document contains false information.

Question:
{question}

Document:
{document_text}
"""

    result = structured_grader.invoke(prompt)

    return result


def grade_documents(question, documents):
    graded_documents = []

    for document in documents:
        grade_result = grade_document(
            question=question,
            document_text=document["text"],
        )

        graded_document = document.copy()
        graded_document["grade"] = grade_result.grade
        graded_document["grade_reason"] = grade_result.reason

        graded_documents.append(graded_document)

    return graded_documents


def choose_crag_route(graded_documents):
    correct_count = 0
    ambiguous_count = 0

    for document in graded_documents:
        if document["grade"] == "correct":
            correct_count = correct_count + 1

        if document["grade"] == "ambiguous":
            ambiguous_count = ambiguous_count + 1

    if correct_count > 0:
        return "generate"

    if ambiguous_count > 0:
        return "rewrite_query"

    return "web_search"


if __name__ == "__main__":
    question = input("Enter a question: ").strip()

    if question == "":
        question = "Which magazine was started first, Arthur's Magazine or First for Women?"

    test_documents = [
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
        {
            "id": "document_3",
            "title": "Basketball",
            "text": "Basketball is a team sport played by two teams on a rectangular court.",
        },
    ]

    graded_documents = grade_documents(
        question=question,
        documents=test_documents,
    )

    print("\nDocument grades:")

    for document in graded_documents:
        print("\nTitle:", document["title"])
        print("Grade:", document["grade"])
        print("Reason:", document["grade_reason"])

    crag_route = choose_crag_route(graded_documents)

    print("\nSelected CRAG route:", crag_route)