def format_docs(docs):
    print(docs)
    return "\n\n".join(doc.page_content for doc in docs)
