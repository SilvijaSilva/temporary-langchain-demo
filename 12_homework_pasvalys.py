# Basic RAG implementation using LangChain and pasvalys.txt
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

from dotenv import load_dotenv

load_dotenv()  # take environment variables

token = os.getenv("MY_TOKEN")  # Replace with your actual token
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1-nano"

# Load the document
loader = TextLoader("documents/pasvalys.txt")
documents = loader.load()

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=token, base_url=endpoint)
db = FAISS.from_documents(docs, embeddings)

# Set up retriever and QA chain
retriever = db.as_retriever()
llm = ChatOpenAI(model=model, base_url=endpoint, api_key=token, temperature=0)
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
