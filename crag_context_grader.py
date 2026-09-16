from typing import Literal

from pydantic import BaseModel, Field

from document_utils import build_context
from models import grader_model


class ContextAssessment(BaseModel):
    status: Literal[
        "sufficient",
        "incomplete",
        "irrelevant",
    ] = Field(
        description=(
            "Whether the combined documents are sufficient, "
            "incomplete, or irrelevant."
        )
    )

    reason: str = Field(
        description="A short explanation of the context decision."
    )

    missing_information: str = Field(
        description=(
            "Information still needed to answer the question. "
            "Use an empty string when the context is sufficient."
        )
    )


structured_context_grader = grader_model.with_structured_output(
    ContextAssessment
)


def assess_context(question, documents):
    if len(documents) == 0:
        return ContextAssessment(
            status="irrelevant",
            reason="No relevant documents were found.",
            missing_information=(
                "All information required to answer the question."
            ),
        )

    context = build_context(documents)

    prompt = f"""
You are the context sufficiency grader in a Corrective RAG system.

Determine whether the combined documents contain enough information
to answer the user's complete question.

The question may require multiple reasoning steps.

Use these statuses:

sufficient:
The documents can be combined to produce a complete, supported answer.

incomplete:
The documents contain useful evidence for part of the question, but
one or more facts or reasoning steps are missing.

irrelevant:
The documents contain no useful evidence for answering the question.

Important rules:

- Judge all documents together.
- Do not require one document to contain the complete answer.
- Follow relationships across multiple documents.
- Do not use outside knowledge.
- Do not answer the question.
- When status is incomplete, clearly describe the missing information.
- When status is sufficient, missing_information must be an empty string.

Question:
{question}

Documents:
{context}
"""

    result = structured_context_grader.invoke(prompt)

    return result