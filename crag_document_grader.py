from typing import Literal

from pydantic import BaseModel, Field

from models import grader_model


class DocumentGrade(BaseModel):
    grade: Literal[
        "correct",
        "ambiguous",
        "incorrect",
    ] = Field(
        description=(
            "Whether the document contributes useful evidence "
            "for answering the question."
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

Determine whether this document provides useful evidence for answering
the user's question.

The question may require multiple reasoning steps.

Use these grades:

correct:
The document explicitly supports at least one fact, entity, or
relationship needed to answer the question.

ambiguous:
The document discusses a related entity or topic, but does not provide
clear evidence for a required reasoning step.

incorrect:
The document is unrelated and provides no useful evidence.

Important rules:

- A correct document does not need to contain the final answer.
- One document may establish the first reasoning step.
- Another document may establish the second reasoning step.
- Do not mark a useful first-hop document as ambiguous only because
  another document is required.
- Do not answer the question.

Example:

If the question asks where the director of a film lives:

- A document identifying the film's director is correct.
- A document stating where that director lives is correct.
- Both documents may be required for the final answer.

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