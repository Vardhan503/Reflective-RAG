from typing import Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


load_dotenv()


class DocumentGrade(BaseModel):
    grade: Literal["correct", "ambiguous", "incorrect"] = Field(
        description="How useful the document is for answering the question."
    )

    reason: str = Field(
        description="A short explanation for the selected grade."
    )


model = ChatOpenAI(model="gpt-5-mini")

structured_grader = model.with_structured_output(DocumentGrade)


def grade_document(question, document_text):
    prompt = f"""
You are a document relevance grader for a Corrective RAG system.

Compare the document with the user's question.

Use one of these grades:

- correct: The document contains information that directly helps answer the question.
- ambiguous: The document is related, but it does not provide enough information to answer confidently.
- incorrect: The document is unrelated or not useful for answering the question.

The word incorrect means irrelevant for this question. It does not necessarily mean
that the document contains false information.

Question:
{question}

Document:
{document_text}
"""

    result = structured_grader.invoke(prompt)

    return result


question = input("Enter a question: ").strip()

if question == "":
    question = "Which magazine was started first, Arthur's Magazine or First for Women?"

document_text = input("Enter a document passage: ").strip()

if document_text == "":
    document_text = """
Arthur's Magazine was an American literary periodical published in Philadelphia
from 1844 to 1846. It was edited by Timothy Shay Arthur.
"""

grade_result = grade_document(question, document_text)

print("\nQuestion:")
print(question)

print("\nDocument:")
print(document_text)

print("\nCRAG grade:", grade_result.grade)
print("Reason:", grade_result.reason)
