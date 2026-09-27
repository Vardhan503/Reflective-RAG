from pydantic import BaseModel, Field

from document_utils import build_context
from models import grader_model


class AnswerCritique(BaseModel):
    useful: bool = Field(
        description=(
            "True when the answer directly and completely "
            "addresses the question."
        )
    )

    needs_more_context: bool = Field(
        description=(
            "True when the answer cannot be completed because "
            "the documents are missing required information."
        )
    )

    reason: str = Field(
        description="A short explanation of the usefulness decision."
    )

    improvement_feedback: str = Field(
        description=(
            "Instructions for improving an answer that is not useful."
        )
    )


structured_critic = grader_model.with_structured_output(
    AnswerCritique
)


def critique_answer(question, answer, documents):
    if len(documents) == 0:
        raise ValueError(
            "At least one document is required to critique an answer."
        )

    context = build_context(documents)

    prompt = f"""
You are the answer critic in a Self-RAG system.

The answer has already passed a hallucination check.

Determine whether the answer completely satisfies the user's question.

Check whether the answer:

- Directly answers the question.
- Addresses every part of the question.
- Includes important information available in the documents.
- Is clear and sufficiently specific.
- Avoids unnecessary information.
- Uses citations appropriately.

Set needs_more_context to true when:

- The answer says there is not enough information, and
- The provided documents genuinely lack a fact needed for the answer.

Set needs_more_context to false when:

- The documents contain the answer, but the generated answer failed
  to use the information properly.
- The answer only needs rewriting or better formatting.
- The answer is already useful.

If useful is true:

- needs_more_context must be false.
- improvement_feedback must be an empty string.

Question:
{question}

Generated answer:
{answer}

Available documents:
{context}
"""

    result = structured_critic.invoke(prompt)

    return result