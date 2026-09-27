from pydantic import BaseModel, Field

from models import generator_model


class RewrittenQuery(BaseModel):
    rewritten_query: str = Field(
        description=(
            "A focused search query for retrieving the missing information."
        )
    )

    reason: str = Field(
        description="A short explanation of how the query was improved."
    )


structured_rewriter = generator_model.with_structured_output(
    RewrittenQuery
)


def rewrite_query(
    question,
    current_query,
    documents,
    missing_information,
):
    document_clues = ""

    for document in documents:
        document_clues = document_clues + "\nTitle: "
        document_clues = document_clues + document["title"]

        document_clues = document_clues + "\nText: "
        document_clues = document_clues + document["text"]

        grade_reason = document.get("grade_reason", "")

        if grade_reason != "":
            document_clues = document_clues + "\nGrader reason: "
            document_clues = document_clues + grade_reason

        document_clues = document_clues + "\n"

    if document_clues == "":
        document_clues = "No useful document clues are available."

    prompt = f"""
You rewrite search queries for a multi-hop Corrective RAG system.

The previous retrieval found useful information, but the combined
documents were not sufficient to answer the complete question.

Write a new search query that focuses on the missing information.

Important rules:

- Use names and entities discovered in the documents.
- Focus on the next missing reasoning step.
- Do not repeat the same query unless no better query is possible.
- Make the new query clear and standalone.
- Do not answer the question.
- Do not add unsupported facts.
- Return one rewritten query.

Original question:
{question}

Previous retrieval query:
{current_query}

Missing information:
{missing_information}

Useful document clues:
{document_clues}
"""

    result = structured_rewriter.invoke(prompt)

    return result