from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


load_dotenv()


class RetrievalDecision(BaseModel):
    retrieve: bool = Field(
        description="True when external documents are needed to handle the request."
    )

    reason: str = Field(
        description="A short explanation of the retrieval decision."
    )


model = ChatOpenAI(model="gpt-5-mini")

structured_router = model.with_structured_output(RetrievalDecision)


def decide_if_retrieval_is_needed(question):
    prompt = f"""
You are the retrieval router in a Self-RAG system.

Decide whether external documents are needed to handle the user's request.

Set retrieve to true for:

- Factual questions about people, places, organizations, events, or history.
- Questions that require private, current, or domain-specific information.
- Questions that ask for evidence, sources, or citations.
- Questions that should be answered using the RAG knowledge base.
- Any request where you are uncertain whether retrieval is needed.

Set retrieve to false only for:

- Greetings and casual conversation that require no factual information.
- Simple arithmetic.
- Rewriting, translating, or formatting text already provided by the user.
- Requests that can be completed entirely from information inside the request.

Do not skip retrieval simply because a language model may already know the answer.

User request:
{question}
"""

    result = structured_router.invoke(prompt)

    return result


if __name__ == "__main__":
    question = input("Enter a question: ").strip()

    if question == "":
        question = "Which magazine was started first, Arthur's Magazine or First for Women?"

    decision = decide_if_retrieval_is_needed(question)

    print("\nQuestion:")
    print(question)

    print("\nRetrieval needed:", decision.retrieve)
    print("Reason:", decision.reason)
