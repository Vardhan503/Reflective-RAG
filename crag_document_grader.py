from typing import Literal

from pydantic import BaseModel, Field

from models import grader_model


class DocumentGrade(BaseModel):
    grade: Literal["correct", "ambiguous", "incorrect"] = Field(
        description=(
            "Whether the document provides useful evidence "
            "for answering at least one part of the question."
        )
    )

    reason: str = Field(
        description="A short explanation for the selected grade."
    )


structured_grader = grader_model.with_structured_output(
    DocumentGrade
)


def grade_document(
    question,
    document_title,
    document_text,
):
    prompt = f"""
You are a document relevance grader for a Corrective RAG system.

Determine whether this document provides evidence that helps answer
the user's question.

Important:

- The question may require multiple reasoning steps.
- One document does not need to contain the complete final answer.
- A document is correct when it supports at least one necessary
  reasoning step.
- Multiple correct documents may need to be combined.
- Grade only the document's usefulness for the question.
- Do not answer the question.

Use these grades:

correct:
The document explicitly provides a fact, entity, or relationship needed
for at least one step of answering the question.

ambiguous:
The document discusses a related entity or topic, but does not provide
clear evidence for a required reasoning step.

incorrect:
The document is unrelated and provides no useful evidence.

Multi-hop example:

If the question requires finding who directed a movie and then finding
where that person lives:

- A document identifying the movie's director is correct.
- A document stating where that director lives is correct.
- Neither document needs to contain the complete answer by itself.

Question:
{question}

Document title:
{document_title}

Document text:
{document_text}
"""

    result = structured_grader.invoke(prompt)

    return result


def grade_documents(question, documents):
    graded_documents = []

    for document in documents:
        grade_result = grade_document(
            question=question,
            document_title=document["title"],
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