from pydantic import BaseModel, Field

from models import generator_model


class RewrittenQuery(BaseModel):
    rewritten_query: str = Field(
        description="A clearer query that can retrieve better documents."
    )

    reason: str = Field(
        description="A short explanation of how the query was improved."
    )


structured_rewriter = generator_model.with_structured_output(RewrittenQuery)


def rewrite_query(question, ambiguous_documents):
    document_clues = ""

    for document in ambiguous_documents:
        document_clues = document_clues + "\nTitle: " + document["title"]
        document_clues = document_clues + "\nText: " + document["text"]
        document_clues = document_clues + "\nGrader reason: " + document["grade_reason"]
        document_clues = document_clues + "\n"

    prompt = f"""
You rewrite questions for a Corrective RAG retrieval system.

The first retrieval returned documents that were related to the question,
but they did not contain enough information to answer confidently.

Rewrite the original question so that the next retrieval has a better chance
of finding the missing information.

Rules:

- Preserve the user's original meaning.
- Make the query clear and standalone.
- Add useful names and keywords found in the document clues.
- Do not answer the question.
- Do not add facts that are not present in the question or document clues.
- Return one rewritten query.

Original question:
{question}

Ambiguous document clues:
{document_clues}
"""

    result = structured_rewriter.invoke(prompt)

    return result
