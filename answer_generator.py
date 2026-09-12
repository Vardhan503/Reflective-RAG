from pydantic import BaseModel, Field

from document_utils import build_context
from models import generator_model


class GeneratedAnswer(BaseModel):
    answer: str = Field(
        description="An answer supported only by the provided documents."
    )

    source_ids: list[str] = Field(
        description="The IDs of the documents used to produce the answer."
    )


structured_generator = generator_model.with_structured_output(GeneratedAnswer)


def generate_answer(question, documents):
    if len(documents) == 0:
        raise ValueError("At least one document is required to generate an answer.")

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

    result = structured_generator.invoke(prompt)

    return result