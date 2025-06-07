# Basic RAG implementation using LangChain and pasvalys.txt
import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from pydantic import BaseModel
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from rag_helpers import format_docs

from dotenv import load_dotenv

load_dotenv()  

token = os.getenv("MY_TOKEN")
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1-nano"

client = ChatOpenAI(
    base_url=endpoint,
    api_key=token,
    model=model,
)

# Load the document
loader = TextLoader("pasvalys.txt")
documents = loader.load()

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    base_url="https://models.inference.ai.azure.com",
    api_key=token
)
db = Chroma.from_documents(docs, embeddings)

# Set up retriever and QA chain
retriever = db.as_retriever()

prompt = hub.pull("rlm/rag-prompt")

def main():
    print("Basic RAG demo about Pasvalys. Type your question (or 'exit' to quit):")
    while True:
        query = input("\nQuestion: ")
        if query.lower() in ("exit", "quit"): break

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | client
            | StrOutputParser()
)

        answer = rag_chain.invoke(query)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main()