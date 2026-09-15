from pydantic import BaseModel, Field

from document_utils import build_context
from models import grader_model


class AnswerCritique(BaseModel):
    useful: bool = Field(
        description="True when the answer directly and completely addresses the question."
    )

    reason: str = Field(
        description="A short explanation of the usefulness decision."
    )

    improvement_feedback: str = Field(
        description="Clear instructions for improving an answer that is not useful."
    )


structured_critic = grader_model.with_structured_output(AnswerCritique)


def critique_answer(question, answer, documents):
    if len(documents) == 0:
        raise ValueError("At least one document is required to critique an answer.")

    context = build_context(documents)

    prompt = f"""
You are the answer critic in a Self-RAG system.

Decide whether the generated answer is useful for the user's question.

The answer has already passed a separate hallucination check. Focus on whether
the answer actually satisfies the user.

Check that the answer:

- Directly answers the question.
- Addresses every part of the question.
- Includes the important information available in the documents.
- Is clear and sufficiently specific.
- Does not contain unnecessary or unrelated information.
- Uses an honest insufficient-information response only when the documents
  truly do not contain enough information.

If the answer is not useful, provide specific instructions for regeneration.
If the answer is useful, set improvement_feedback to an empty string.

Question:
{question}

Generated answer:
{answer}

Available documents:
{context}
"""

    result = structured_critic.invoke(prompt)

    return result
