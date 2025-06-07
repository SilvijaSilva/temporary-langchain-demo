TITLE: Install LangChain via pip
DESCRIPTION: Installs the core LangChain library using the Python package installer, pip. This is the standard and simplest way to get started with LangChain in a Python environment.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/libs/langchain/README.md#_snippet_0

LANGUAGE: Shell
CODE:
```
pip install langchain
```

----------------------------------------

TITLE: Load and Split Documents for Embedding (Python)
DESCRIPTION: Imports necessary loaders and splitters, loads a text document (`state_of_the_union.txt`), splits it into smaller chunks using `CharacterTextSplitter`, and initializes `OpenAIEmbeddings`. This prepares the text data and the embedding model for creating vector representations.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/vectorstores/duckdb.ipynb#_snippet_3

LANGUAGE: python
CODE:
```
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()

documents = CharacterTextSplitter().split_documents(documents)
embeddings = OpenAIEmbeddings()
```

----------------------------------------

TITLE: Loading Documents into Yellowbrick Vector Store (Python)
DESCRIPTION: Defines parameters for document splitting. Creates `Document` objects from previously extracted Yellowbrick data, mapping path and content. Uses `RecursiveCharacterTextSplitter` to split documents into smaller chunks based on specified separators and size limits. Generates embeddings for the split documents using `OpenAIEmbeddings`. Initializes and populates a `Yellowbrick` vector store instance from the split documents and embeddings, connecting to the specified database and table.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/vectorstores/yellowbrick.ipynb#_snippet_6

LANGUAGE: python
CODE:
```
# Split documents into chunks for conversion to embeddings
DOCUMENT_BASE_URL = "https://docs.yellowbrick.com/6.7.1/"  # Actual URL


separator = "\n## "  # This separator assumes Markdown docs from the repo uses ### as logical main header most of the time
chunk_size_limit = 2000
max_chunk_overlap = 200

documents = [
    Document(
        page_content=document[1],
        metadata={"source": DOCUMENT_BASE_URL + document[0].replace(".md", ".html")},
    )
    for document in yellowbrick_documents
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size_limit,
    chunk_overlap=max_chunk_overlap,
    separators=[separator, "\nn", "\n", ",", " ", ""],
)
split_docs = text_splitter.split_documents(documents)

docs_text = [doc.page_content for doc in split_docs]

embeddings = OpenAIEmbeddings()
vector_store = Yellowbrick.from_documents(
    documents=split_docs,
    embedding=embeddings,
    connection_string=yellowbrick_connection_string,
    table=embedding_table,
)

print(f"Created vector store with {len(documents)} documents")
```

----------------------------------------

TITLE: Set OpenAI API Key - Python
DESCRIPTION: Sets the OpenAI API key as an environment variable, which is required for using `OpenAIEmbeddings` and `OpenAI` LLM. Replace the placeholder with your actual key.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/retrievers/self_query/mongodb_atlas.ipynb#_snippet_1

LANGUAGE: python
CODE:
```
import os

OPENAI_API_KEY = "Use your OpenAI key"

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
```

----------------------------------------

TITLE: Pass Base64 PDF to LangChain Chat Model
DESCRIPTION: Demonstrates fetching a PDF from a URL, encoding it to base64, initializing a LangChain chat model, and sending a message containing text and the base64 PDF data.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/how_to/multimodal_inputs.ipynb#_snippet_7

LANGUAGE: python
CODE:
```
import base64

import httpx
from langchain.chat_models import init_chat_model

# Fetch PDF data
pdf_url = "https://pdfobject.com/pdf/sample.pdf"
pdf_data = base64.b64encode(httpx.get(pdf_url).content).decode("utf-8")


# Pass to LLM
llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")

message = {
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": "Describe the document:",
        },
        # highlight-start
        {
            "type": "file",
            "source_type": "base64",
            "data": pdf_data,
            "mime_type": "application/pdf",
        },
        # highlight-end
    ],
}
response = llm.invoke([message])
print(response.text())
```

----------------------------------------

TITLE: Constructing and Invoking RAG Chain - Python
DESCRIPTION: Imports necessary components (StrOutputParser, RunnablePassthrough) and constructs a RAG chain using LangChain's LCEL (LangChain Expression Language). The chain defines the flow: retrieve context, pass context and question to the prompt, send to the LLM, and parse the output as a string. Finally, it invokes the chain with a specific question.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/vectorstores/weaviate.ipynb#_snippet_24

LANGUAGE: python
CODE:
```
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("What did the president say about Justice Breyer")
```

----------------------------------------

TITLE: Initialize Langgraph StateGraph with Memory
DESCRIPTION: Sets up a basic Langgraph `StateGraph` for conversational AI. It defines a node to call the language model and integrates `MemorySaver` for persistence, compiling the workflow into an application.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/versions/migrating_chains/conversation_chain.ipynb#_snippet_4

LANGUAGE: Python
CODE:
```
import uuid

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

model = ChatOpenAI(model="gpt-4o-mini")

# Define a new graph
workflow = StateGraph(state_schema=MessagesState)


# Define the function that calls the model
def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}


# Define the two nodes we will cycle between
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


# The thread id is a unique key that identifies
# this particular conversation.
# We'll just generate a random uuid here.
thread_id = uuid.uuid4()
config = {"configurable": {"thread_id": thread_id}}
```

----------------------------------------

TITLE: Create ReAct Agent with LangGraph
DESCRIPTION: Initializes a ReAct agent executor using LangGraph's prebuilt function. This sets up the agent's control flow, combining the language model and tools to decide actions.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/tutorials/agents.ipynb#_snippet_14

LANGUAGE: python
CODE:
```
from langgraph.prebuilt import create_react_agent

agent_executor = create_react_agent(model, tools)
```

----------------------------------------

TITLE: Building RAG Chain with Langchain Expression Language - Python
DESCRIPTION: Imports `RunnablePassthrough` for passing input through the chain. Constructs a RAG chain using the Langchain Expression Language (`|`). The chain takes a question, retrieves context using the `retriever`, passes both to the `prompt`, and finally uses the `hf` LLM to generate the answer.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/cookbook/rag_with_quantized_embeddings.ipynb#_snippet_10

LANGUAGE: python
CODE:
```
from langchain.schema.runnable import RunnablePassthrough

rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | hf
```

----------------------------------------

TITLE: Installing LangChain Library - Python
DESCRIPTION: Demonstrates how to install or upgrade the LangChain library using pip, a common prerequisite for running LangChain examples and integrations.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/chat/nvidia_ai_endpoints.ipynb#_snippet_18

LANGUAGE: Python
CODE:
```
%pip install --upgrade --quiet langchain
```

----------------------------------------

TITLE: Creating a React Agent with LangGraph (Python)
DESCRIPTION: This code creates a `react_agent` using `langgraph.prebuilt.create_react_agent`. It takes an LLM instance and a list of tools (presumably from `toolkit.get_tools()`) to enable the agent to reason and interact with the Memgraph database.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/tools/memgraph.ipynb#_snippet_5

LANGUAGE: python
CODE:
```
from langgraph.prebuilt import create_react_agent

agent_executor = create_react_agent(llm, tools)
```

----------------------------------------

TITLE: Defining Retrieval Grader for Document Relevance (Python)
DESCRIPTION: This snippet creates a `retrieval_grader` to assess the relevance of retrieved documents to a user's question. It uses a `PromptTemplate` to instruct an LLM to provide a binary 'yes' or 'no' score, parsed as JSON. This grader helps filter out irrelevant documents, improving the quality of the RAG system's responses.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/cookbook/local_rag_agents_intel_cpu.ipynb#_snippet_7

LANGUAGE: python
CODE:
```
from langchain_core.output_parsers import JsonOutputParser

prompt = PromptTemplate(
    template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n
    If the document contains keywords related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.""",
    input_variables=["question", "document"],
)

retrieval_grader = prompt | llm | JsonOutputParser()
```

----------------------------------------

TITLE: Run LangGraph with Constitutional Principles - Python
DESCRIPTION: This snippet demonstrates how to execute the previously defined LangGraph. It sets up a list of `ConstitutionalPrinciple` objects and a query, then invokes the compiled graph asynchronously, streaming the intermediate steps and printing relevant parts of the state dictionary.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/versions/migrating_chains/constitutional_chain.ipynb#_snippet_5

LANGUAGE: python
CODE:
```
constitutional_principles = [
    ConstitutionalPrinciple(
        critique_request="Tell if this answer is good.",
        revision_request="Give a better answer.",
    )
]

query = "What is the meaning of life? Answer in 10 words or fewer."

async for step in app.astream(
    {"query": query, "constitutional_principles": constitutional_principles},
    stream_mode="values",
):
    subset = ["initial_response", "critiques_and_revisions", "response"]
    print({k: v for k, v in step.items() if k in subset})
```

----------------------------------------

TITLE: Creating LangChain Function Calling Chain (Python)
DESCRIPTION: Constructs a LangChain processing chain. It uses a chat prompt, binds the `Calculator` function definition to an OpenAI chat model, parses the model's output using `PydanticOutputFunctionsParser`, and finally applies a lambda function to execute the calculation.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/chat_loaders/langsmith_llm_runs.ipynb#_snippet_4

LANGUAGE: Python
CODE:
```
from langchain_core.output_parsers.openai_functions import PydanticOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an accounting assistant."),
        ("user", "{input}"),
    ]
)
chain = (
    prompt
    | ChatOpenAI().bind(functions=[openai_function_def])
    | PydanticOutputFunctionsParser(pydantic_schema=Calculator)
    | (lambda x: x.calculate())
)
```

----------------------------------------

TITLE: Parse JSON Output with Pydantic (Python)
DESCRIPTION: Demonstrates how to use JsonOutputParser with a Pydantic model to define the expected JSON schema. It sets up a chat model, defines the data structure, creates a parser linked to the Pydantic object, builds a prompt template including format instructions, chains the components, and invokes the chain to get structured output.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/how_to/output_parser_json.ipynb#_snippet_1

LANGUAGE: python
CODE:
```
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

model = ChatOpenAI(temperature=0)


# Define your desired data structure.
class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")


# And a query intented to prompt a language model to populate the data structure.
joke_query = "Tell me a joke."

# Set up a parser + inject instructions into the prompt template.
parser = JsonOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | model | parser

chain.invoke({"query": joke_query})
```

----------------------------------------

TITLE: Using LangChain with_structured_output for Schema Binding (Python)
DESCRIPTION: Illustrates how to use LangChain's `with_structured_output` helper to bind a specific schema (like a Pydantic object) to the model, enabling automatic parsing of the model's output into the defined structure.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/concepts/structured_outputs.mdx#_snippet_6

LANGUAGE: python
CODE:
```
# Bind the schema to the model
model_with_structure = model.with_structured_output(ResponseFormatter)
# Invoke the model
structured_output = model_with_structure.invoke("What is the powerhouse of the cell?")
# Get back the pydantic object
structured_output
ResponseFormatter(answer="The powerhouse of the cell is the mitochondrion. Mitochondria are organelles that generate most of the cell's supply of adenosine triphosphate (ATP), which is used as a source of chemical energy.", followup_question='What is the function of ATP in the cell?')

```

----------------------------------------

TITLE: Creating a LangChain Retrieval Chain with OpenAI
DESCRIPTION: This snippet defines an LCEL chain that integrates a retriever, a chat prompt, an OpenAI model, and a string output parser. It sets up a system prompt for a software engineer persona, providing context from LangChain documentation to answer user questions or generate code. The chain processes a question, retrieves relevant context, and generates an answer.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/retrievers/fleet_context.ipynb#_snippet_6

LANGUAGE: python
CODE:
```
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a great software engineer who is very familiar \
with Python. Given a user question or request about a new Python library called LangChain and \
parts of the LangChain documentation, answer the question or generate the requested code. \
Your answers must be accurate, should include code whenever possible, and should assume anything \
about LangChain which is note explicitly stated in the LangChain documentation. If the required \
information is not available, just say so.

LangChain Documentation
------------------

{context}""",
        ),
        ("human", "{question}"),
    ]
)

model = ChatOpenAI(model="gpt-3.5-turbo-16k")

chain = (
    {
        "question": RunnablePassthrough(),
        "context": parent_retriever
        | (lambda docs: "\n\n".join(d.page_content for d in docs)),
    }
    | prompt
    | model
    | StrOutputParser()
)
```

----------------------------------------

TITLE: Install LangChain OpenAI Integration Package (Shell)
DESCRIPTION: Installs the `langchain-openai` Python package using pip, which provides the necessary classes and functions to interact with OpenAI models via LangChain.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/llms/openai.ipynb#_snippet_2

LANGUAGE: shell
CODE:
```
%pip install -qU langchain-openai
```

----------------------------------------

TITLE: Installing LangChain OpenAI Integration (Shell)
DESCRIPTION: Executes a shell command using `%pip` (common in Jupyter notebooks) to install or upgrade the `langchain-openai` package. This package provides the necessary integration components for using LangChain with OpenAI services, including fine-tuning.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/chat_loaders/imessage.ipynb#_snippet_7

LANGUAGE: Shell
CODE:
```
%pip install --upgrade --quiet  langchain-openai
```

----------------------------------------

TITLE: Defining and Binding Tools to LangChain Model (Python)
DESCRIPTION: Defines 'add' and 'multiply' functions as LangChain tools using the @tool decorator, providing docstrings as descriptions. It then creates a list of these tools and binds them to the llm instance, creating a new model instance capable of using these tools.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/how_to/tool_results_pass_to_model.ipynb#_snippet_1

LANGUAGE: python
CODE:
```
from langchain_core.tools import tool\n\n\n@tool\ndef add(a: int, b: int) -> int:\n    """Adds a and b."""\n    return a + b\n\n\n@tool\ndef multiply(a: int, b: int) -> int:\n    """Multiplies a and b."""\n    return a * b\n\n\ntools = [add, multiply]\n\nllm_with_tools = llm.bind_tools(tools)
```

----------------------------------------

TITLE: Constructing and Compiling LangGraph for Summarization in Python
DESCRIPTION: This snippet demonstrates how to construct a 'StateGraph' using LangGraph. It defines several nodes ('generate_summary', 'collect_summaries', 'collapse_summaries', 'generate_final_summary') and establishes conditional and direct edges between them, forming a workflow for document summarization. Finally, the graph is compiled into an executable application.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/tutorials/summarization.ipynb#_snippet_23

LANGUAGE: python
CODE:
```
graph = StateGraph(OverallState)
graph.add_node("generate_summary", generate_summary)  # same as before
graph.add_node("collect_summaries", collect_summaries)
graph.add_node("collapse_summaries", collapse_summaries)
graph.add_node("generate_final_summary", generate_final_summary)

# Edges:
graph.add_conditional_edges(START, map_summaries, ["generate_summary"])
graph.add_edge("generate_summary", "collect_summaries")
graph.add_conditional_edges("collect_summaries", should_collapse)
graph.add_conditional_edges("collapse_summaries", should_collapse)
graph.add_edge("generate_final_summary", END)

app = graph.compile()
```

----------------------------------------

TITLE: Filtering Stream Events by Tags - Langchain Python
DESCRIPTION: This example demonstrates filtering `astream_events` using component tags via the `include_tags` parameter. It applies a tag 'my_chain' to the entire chain and then requests only events associated with this tag.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/how_to/streaming.ipynb#_snippet_21

LANGUAGE: Python
CODE:
```
chain = (model | JsonOutputParser()).with_config({"tags": ["my_chain"]})

max_events = 0
async for event in chain.astream_events(
    'output a list of the countries france, spain and japan and their populations in JSON format. Use a dict with an outer key of "countries" which contains a list of countries. Each country should have the key `name` and `population`',
    include_tags=["my_chain"],
):
    print(event)
    max_events += 1
    if max_events > 10:
        # Truncate output
        print("...")
        break
```

----------------------------------------

TITLE: Implementing RAG with Langchain and JaguarDB
DESCRIPTION: This snippet shows how to build a Retrieval Augmented Generation (RAG) system using Langchain components integrated with a Jaguar Vector Database. It covers loading and processing documents, initializing the Jaguar vector store, adding data, configuring a retriever, setting up a prompt template, using an OpenAI LLM, and executing the RAG chain to answer a question based on retrieved context.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/vectorstores/jaguar.ipynb#_snippet_1

LANGUAGE: python
CODE:
```
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.jaguar import Jaguar
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

""" 
Load a text file into a set of documents 
"""
loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
docs = text_splitter.split_documents(documents)

"""
Instantiate a Jaguar vector store
"""
### Jaguar HTTP endpoint
url = "http://192.168.5.88:8080/fwww/"

### Use OpenAI embedding model
embeddings = OpenAIEmbeddings()

### Pod is a database for vectors
pod = "vdb"

### Vector store name
store = "langchain_rag_store"

### Vector index name
vector_index = "v"

### Type of the vector index
# cosine: distance metric
# fraction: embedding vectors are decimal numbers
# float: values stored with floating-point numbers
vector_type = "cosine_fraction_float"

### Dimension of each embedding vector
vector_dimension = 1536

### Instantiate a Jaguar store object
vectorstore = Jaguar(
    pod, store, vector_index, vector_type, vector_dimension, url, embeddings
)

"""
Login must be performed to authorize the client.
The environment variable JAGUAR_API_KEY or file $HOME/.jagrc
should contain the API key for accessing JaguarDB servers.
"""
vectorstore.login()


"""
Create vector store on the JaguarDB database server.
This should be done only once.
"""
# Extra metadata fields for the vector store
metadata = "category char(16)"

# Number of characters for the text field of the store
text_size = 4096

#  Create a vector store on the server
vectorstore.create(metadata, text_size)

"""
Add the texts from the text splitter to our vectorstore
"""
vectorstore.add_documents(docs)
# or tag the documents:
# vectorstore.add_documents(more_docs, text_tag="tags to these documents")

""" Get the retriever object """
retriever = vectorstore.as_retriever()
# retriever = vectorstore.as_retriever(search_kwargs={"where": "m1='123' and m2='abc'"})

template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

""" Obtain a Large Language Model """
LLM = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

""" Create a chain for the RAG flow """
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | LLM
    | StrOutputParser()
)

resp = rag_chain.invoke("What did the president say about Justice Breyer?")
print(resp)
```

----------------------------------------

TITLE: Streaming LLM Events Asynchronously with Langchain
DESCRIPTION: Illustrates how to stream events from an LLM using the `astream_events` method in Langchain. This method is useful for complex applications involving multiple steps. The example iterates through events, prints them, and includes a simple truncation logic. Requires `langchain_openai`.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/how_to/streaming_llm.ipynb#_snippet_2

LANGUAGE: python
CODE:
```
from langchain_openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0, max_tokens=512)

idx = 0

async for event in llm.astream_events(
    "Write me a 1 verse song about goldfish on the moon", version="v1"
):
    idx += 1
    if idx >= 5:  # Truncate the output
        print("...Truncated")
        break
    print(event)
```

----------------------------------------

TITLE: Creating ChatPromptTemplate for Translation (Python)
DESCRIPTION: Defines a `ChatPromptTemplate` using `langchain_core.prompts` with system and human message templates that accept input and output language variables.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/chat/ibm_watsonx.ipynb#_snippet_10

LANGUAGE: python
CODE:
```
from langchain_core.prompts import ChatPromptTemplate

system = (
    "You are a helpful assistant that translates {input_language} to {output_language}."
)
human = "{input}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
```

----------------------------------------

TITLE: Evaluate Embedding + BM25 Retriever with Reranker in Python
DESCRIPTION: Demonstrates calling the `evaluate` function to assess the performance of a specific retriever configuration: an embedding and BM25 hybrid retriever combined with a reranker. It uses the `qa_pairs` dataset for evaluation.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/cookbook/contextual_rag.ipynb#_snippet_19

LANGUAGE: python
CODE:
```
embedding_bm25_rerank_results = evaluate(embedding_bm25_retriever_rerank, qa_pairs)
```

----------------------------------------

TITLE: Setting __ModuleName__ API Key Environment Variable
DESCRIPTION: This snippet demonstrates how to securely set the __ModuleName__ API key as an environment variable. It checks if the `__MODULE_NAME___API_KEY` is already set and, if not, prompts the user to enter it using `getpass` for security. This is a prerequisite for authenticating with __ModuleName__ services.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/libs/cli/langchain_cli/integration_template/docs/chat.ipynb#_snippet_0

LANGUAGE: python
CODE:
```
import getpass
import os

if not os.getenv("__MODULE_NAME___API_KEY"):
    os.environ["__MODULE_NAME___API_KEY"] = getpass.getpass("Enter your __ModuleName__ API key: ")
```

----------------------------------------

TITLE: Defining Structured Output Schema with Pydantic (Python)
DESCRIPTION: This snippet demonstrates how to define a `Joke` Pydantic BaseModel to enforce a specific structured output format for LLM responses, including fields for setup and punchline. It then shows how to configure an LLM instance to produce outputs conforming to this schema using `llm.with_structured_output()`, and invokes it to generate a joke.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/chat/sambanova.ipynb#_snippet_15

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, Field


class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")


structured_llm = llm.with_structured_output(Joke)

structured_llm.invoke("Tell me a joke about cats")
```

----------------------------------------

TITLE: Invoke Hybrid Query Combining Two Semantic Searches (LangChain)
DESCRIPTION: Shows a complex hybrid query using `full_chain.invoke` that combines two semantic search criteria: finding songs about "breakouts" from albums semantically related to "love".
SOURCE: https://github.com/langchain-ai/langchain/blob/master/cookbook/retrieval_in_sql.ipynb#_snippet_20

LANGUAGE: python
CODE:
```
full_chain.invoke(
    {
        "question": "I want to know songs about breakouts obtained from top 5 albums about love"
    }
)
```

----------------------------------------

TITLE: Define and Invoke Basic LangChain Runnable (Python)
DESCRIPTION: Sets up a basic LangChain runnable sequence using LCEL. It includes a chat prompt template, a ChatOpenAI model, and a string output parser, demonstrating how to invoke the sequence with a simple input.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/how_to/binding.ipynb#_snippet_1

LANGUAGE: python
CODE:
```
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Write out the following equation using algebraic symbols then solve it. Use the format\n\nEQUATION:...\nSOLUTION:...\n\n",
        ),
        ("human", "{equation_statement}"),
    ]
)

model = ChatOpenAI(temperature=0)

runnable = (
    {"equation_statement": RunnablePassthrough()} | prompt | model | StrOutputParser()
)

print(runnable.invoke("x raised to the third plus seven equals 12"))
```

----------------------------------------

TITLE: Creating Document with Metadata - Langchain - Python
DESCRIPTION: Constructs a Langchain Document object, incorporating both the main text content from the 'text' variable and the contextual information stored in the 'metadata' dictionary. This allows for richer document representation.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/document_loaders/copypaste.ipynb#_snippet_4

LANGUAGE: python
CODE:
```
doc = Document(page_content=text, metadata=metadata)
```

----------------------------------------

TITLE: Creating 'Reasoning' Chain LangChain
DESCRIPTION: Constructs a LangChain chain for the 'reasoning' step by piping the 'reasoning_prompt' to the language 'model' and parsing the output as a string.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/cookbook/self-discover.ipynb#_snippet_16

LANGUAGE: python
CODE:
```
reasoning_chain = reasoning_prompt | model | StrOutputParser()
```

----------------------------------------

TITLE: Defining Structured Output Schema with Pydantic (Python)
DESCRIPTION: Shows how to define a structured output schema using Pydantic's BaseModel and Field, which provides type hints and validation for the model's response.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/concepts/structured_outputs.mdx#_snippet_2

LANGUAGE: python
CODE:
```
from pydantic import BaseModel, Field
class ResponseFormatter(BaseModel):
    """Always use this tool to structure your response to the user."""
    answer: str = Field(description="The answer to the user's question")
    followup_question: str = Field(description="A followup question the user could ask")
```

----------------------------------------

TITLE: Using LangGraph Pre-built React Agent with Tool (Python)
DESCRIPTION: Provides a complete example demonstrating the use of `langgraph.prebuilt.create_react_agent`. It includes necessary imports, defines a sample tool (`get_user_age`), a prompt function, creates the agent with the tool and memory, configures it with a thread ID, and runs a sample conversation demonstrating tool usage and memory.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/versions/migrating_memory/conversation_buffer_window_memory.ipynb#_snippet_12

LANGUAGE: Python
CODE:
```
import uuid

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent


@tool
def get_user_age(name: str) -> str:
    """Use this tool to find the user's age."""
    # This is a placeholder for the actual implementation
    if "bob" in name.lower():
        return "42 years old"
    return "41 years old"


memory = MemorySaver()
model = ChatOpenAI()


# highlight-start
def prompt(state) -> list[BaseMessage]:
    """Given the agent state, return a list of messages for the chat model."""
    # We're using the message processor defined above.
    return trim_messages(
        state["messages"],
        token_counter=len,  # <-- len will simply count the number of messages rather than tokens
        max_tokens=5,  # <-- allow up to 5 messages.
        strategy="last",
        # Most chat models expect that chat history starts with either:
        # (1) a HumanMessage or
        # (2) a SystemMessage followed by a HumanMessage
        # start_on="human" makes sure we produce a valid chat history
        start_on="human",
        # Usually, we want to keep the SystemMessage
        # if it's present in the original history.
        # The SystemMessage has special instructions for the model.
        include_system=True,
        allow_partial=False,
    )


# highlight-end

app = create_react_agent(
    model,
    tools=[get_user_age],
    checkpointer=memory,
    # highlight-next-line
    prompt=prompt,
)

# The thread id is a unique key that identifies
# this particular conversation.
# We'll just generate a random uuid here.
thread_id = uuid.uuid4()
config = {"configurable": {"thread_id": thread_id}}

# Tell the AI that our name is Bob, and ask it to use a tool to confirm
# that it's capable of working like an agent.
input_message = HumanMessage(content="hi! I'm bob. What is my age?")

for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()
```

----------------------------------------

TITLE: Defining Nodes and Building LangGraph for Map-Reduce QA (Python)
DESCRIPTION: Defines the state structure (`MapState`), the map function (`map_analyses`) to distribute tasks, the analysis function (`generate_analysis`) to process individual documents, and the reduce function (`pick_top_ranked`) to select the best result. It then constructs the LangGraph using these components, adding nodes and defining conditional and direct edges.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/versions/migrating_chains/map_rerank_docs_chain.ipynb#_snippet_5

LANGUAGE: python
CODE:
```
# This will be the state of the node that we will "map" all
# documents to in order to generate answers with scores
class MapState(TypedDict):
    content: str


# Here we define the logic to map out over the documents
# We will use this an edge in the graph
def map_analyses(state: State):
    # We will return a list of `Send` objects
    # Each `Send` object consists of the name of a node in the graph
    # as well as the state to send to that node
    return [
        Send("generate_analysis", {"content": content}) for content in state["contents"]
    ]


# Here we generate an answer with score, given a document
async def generate_analysis(state: MapState):
    response = await map_chain.ainvoke(state["content"])
    return {"answers_with_scores": [response]}


# Here we will select the top answer
def pick_top_ranked(state: State):
    ranked_answers = sorted(
        state["answers_with_scores"], key=lambda x: -int(x["score"])
    )
    return {"answer": ranked_answers[0]}


# Construct the graph: here we put everything together to construct our graph
graph = StateGraph(State)
graph.add_node("generate_analysis", generate_analysis)
graph.add_node("pick_top_ranked", pick_top_ranked)
graph.add_conditional_edges(START, map_analyses, ["generate_analysis"])
graph.add_edge("generate_analysis", "pick_top_ranked")
graph.add_edge("pick_top_ranked", END)
app = graph.compile()
```

----------------------------------------

TITLE: Compiling LangGraph Application (Python)
DESCRIPTION: This Python code block demonstrates how to compile a LangGraph application. It adds defined nodes ('query_or_respond', 'tools', 'generate') to a graph builder, sets the entry point, defines conditional edges based on a 'tools_condition' to either end the graph or proceed to 'tools', and adds sequential edges from 'tools' to 'generate' and from 'generate' to the END.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/tutorials/qa_chat_history.ipynb#_snippet_11

LANGUAGE: python
CODE:
```
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition

graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile()
```

----------------------------------------

TITLE: Load and Split Documents, Initialize Embeddings and Connection Details
DESCRIPTION: Loads text from a file, splits it into chunks using a CharacterTextSplitter, initializes the OpenAIEmbeddings model, and retrieves the database connection string and collection name.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/vectorstores/pgembedding.ipynb#_snippet_6

LANGUAGE: python
CODE:
```
loader = TextLoader("state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
connection_string = os.environ.get("DATABASE_URL")
collection_name = "state_of_the_union"
```

----------------------------------------

TITLE: Basic RAG Workflow with Langchain Python
DESCRIPTION: Demonstrates a simple RAG pipeline using Langchain. It defines a system prompt to guide the model, retrieves relevant documents based on a user question, formats the system prompt with the retrieved context, initializes a ChatOpenAI model, and invokes the model with the formatted prompt and the user question to generate an answer grounded in the retrieved information. Requires a pre-configured `retriever` object.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/concepts/rag.mdx#_snippet_0

LANGUAGE: python
CODE:
```
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Define a system prompt that tells the model how to use the retrieved context
system_prompt = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.
Context: {context}:"""
    
# Define a question
question = """What are the main components of an LLM-powered autonomous agent system?"""

# Retrieve relevant documents
docs = retriever.invoke(question)

# Combine the documents into a single string
docs_text = "".join(d.page_content for d in docs)

# Populate the system prompt with the retrieved context
system_prompt_fmt = system_prompt.format(context=docs_text)

# Create a model
model = ChatOpenAI(model="gpt-4o", temperature=0) 

# Generate a response
questions = model.invoke([SystemMessage(content=system_prompt_fmt),
                          HumanMessage(content=question)])
```

----------------------------------------

TITLE: Streaming LangGraph with Conversational Input (Python)
DESCRIPTION: This Python code demonstrates streaming the execution of the compiled LangGraph with a simple conversational input ('Hello'). It iterates through the steps produced by the graph's stream method and prints the last message from each step using `pretty_print()`, showing the application's response to a non-retrieval query.
SOURCE: https://github.com/langchain-ai/langchain/blob/master/docs/docs/tutorials/qa_chat_history.ipynb#_snippet_13

LANGUAGE: python
CODE:
```
input_message = "Hello"

for step in graph.stream(
    {"messages": [{"role": "user", "content": input_message}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
```