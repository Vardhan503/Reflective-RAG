from pydantic import BaseModel, Field

from document_utils import build_context
from models import grader_model


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


structured_checker = grader_model.with_structured_output(HallucinationResult)


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
