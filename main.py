# Basic RAG implementation using LangChain and pasvalys.txt
import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Set your OpenAI API key (use env variable for security)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "sk-...your-key...")

# Load the document
loader = TextLoader("documents/pasvalys.txt")
documents = loader.load()

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)

# Set up retriever and QA chain
retriever = db.as_retriever()
llm = OpenAI(temperature=0)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def main():
    print("Basic RAG demo about Pasvalys. Type your question (or 'exit' to quit):")
    while True:
        query = input("\nQuestion: ")
        if query.lower() in ("exit", "quit"): break
        answer = qa.run(query)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
