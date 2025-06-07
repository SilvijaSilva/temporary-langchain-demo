from haystack import Pipeline, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import DocumentEmbedder
from haystack.components.retrievers import MemoryRetriever
from haystack.components.generators import PromptNode

# 1. Load the document
with open("pasvalys.txt", encoding="utf-8") as f:
    text = f.read()

haystack_doc = Document(content=text)

# 2. Set up the document store and add the document
store = InMemoryDocumentStore()
store.write_documents([haystack_doc])

# 3. Embed documents
embedder = DocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
embedder.warm_up()
embedder.run(store)

# 4. Set up the retriever
retriever = MemoryRetriever(document_store=store, retrieval_method="embedding")

# 5. Set up the generator
prompt_node = PromptNode(model_name_or_path="google/flan-t5-base", max_length=256)

# 6. Build the pipeline
pipe = Pipeline()
pipe.add_node(component=retriever, name="Retriever", inputs=["Query"])
pipe.add_node(component=prompt_node, name="PromptNode", inputs=["Retriever"])

def main():
    print("Simple Haystack RAG demo on pasvalys.txt. Type your question (or 'exit' to quit):")
    while True:
        query = input("\nQuestion: ")
        if query.lower() in ("exit", "quit"): break
        result = pipe.run({"Query": query})
        answers = result["PromptNode"].get("answers", [])
        answer = answers[0].answer if answers else "No answer found."
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
