from pydantic import BaseModel, Field

from document_utils import build_context
from models import grader_model


class HallucinationResult(BaseModel):
    grounded: bool = Field(
        description=(
            "True when every factual claim in the answer "
            "is supported by the documents."
        )
    )

    reason: str = Field(
        description="A short explanation of the grounding decision."
    )

    unsupported_claims: list[str] = Field(
        description=(
            "Unsupported claims taken only from the generated answer."
        )
    )


structured_checker = grader_model.with_structured_output(
    HallucinationResult
)


def check_hallucination(question, answer, documents):
    if len(documents) == 0:
        raise ValueError(
            "At least one document is required for grounding verification."
        )

    context = build_context(documents)

    prompt = f"""
You are the hallucination checker in a RAG system.

Determine whether the factual claims in the generated answer are
supported by the provided documents.

Rules:

- Evaluate only claims made in the generated answer.
- The user's question is not a factual claim.
- Never copy the user's question into unsupported_claims.
- Interpret short answers as responses to the question.
- Evidence may be combined across multiple documents.
- Follow relationships between entities across documents.
- Equivalent wording does not need to be an exact quotation.
- Source IDs alone are not evidence.
- Do not use outside knowledge.
- unsupported_claims must only contain statements from the answer.
- The question describes what must be verified, but the question itself
  is not evidence.
- For a multi-hop answer, verify every relationship in the reasoning chain.
- A document saying Person A is a film director does not prove that
  Person A directed Film B.
- The documents must support both the intermediate entity and the final answer.

Insufficient-information rules:

- Saying that the provided documents do not contain enough information
  is not a hallucinated real-world claim.
- If the documents genuinely lack the requested information, an honest
  insufficient-information response is grounded.
- If the answer makes no unsupported factual claims, grounded should
  be true.
- The answer critic will separately decide whether the answer is useful.

Multi-document example:

If one document says Person A directed Film B and another document says
Person A is based in Place C, then "Place C" is supported as the answer
to where the director of Film B is based.

Question:
{question}

Generated answer:
{answer}

Documents:
{context}
"""

    result = structured_checker.invoke(prompt)

    return result