import os

from dotenv import load_dotenv
from tavily import TavilyClient


load_dotenv()


def search_web(question, max_results=5):
    api_key = os.getenv("TAVILY_API_KEY")

    if api_key is None or api_key.strip() == "":
        raise ValueError("TAVILY_API_KEY was not found in the .env file.")

    client = TavilyClient(api_key=api_key)

    response = client.search(
        query=question,
        search_depth="advanced",
        max_results=max_results,
        include_answer=False,
        include_raw_content=False,
    )

    search_results = response.get("results", [])
    web_documents = []

    for result_number in range(len(search_results)):
        search_result = search_results[result_number]
        
        print(f'Search result: {search_result}')
        document = {
            "id": "web_" + str(result_number + 1),
            "title": search_result.get("title", "Untitled web page"),
            "text": search_result.get("content", ""),
            "url": search_result.get("url", ""),
            "source": "web",
        }

        web_documents.append(document)

    return web_documents


if __name__ == "__main__":
    question = input("Enter a question: ").strip()

    if question == "":
        question = "Which magazine was started first, Arthur's Magazine or First for Women?"

    documents = search_web(
        question=question,
        max_results=5,
    )

    print("\nWeb documents found:", len(documents))

    for document in documents:
        print("\nTitle:", document["title"])
        print("URL:", document["url"])
        print("Text:", document["text"][:300])
