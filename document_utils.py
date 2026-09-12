def build_context(documents):
    context = ""

    for document in documents:
        context = context + "\nSource ID: " + document["id"]
        context = context + "\nTitle: " + document["title"]
        context = context + "\nText: " + document["text"]

        source = document.get("source", "")

        if source != "":
            context = context + "\nSource type: " + source

        url = document.get("url", "")

        if url != "":
            context = context + "\nURL: " + url

        context = context + "\n----------------------------------------\n"

    return context.strip()