TITLE: Install Haystack via Pip
DESCRIPTION: This command installs the latest stable version of the Haystack library using the pip package manager. It is the standard and simplest way to get started with Haystack.
SOURCE: https://github.com/deepset-ai/haystack/blob/main/README.md#_snippet_0

LANGUAGE: sh
CODE:
```
pip install haystack-ai
```

----------------------------------------

TITLE: Programmatic Agent Creation in Haystack
DESCRIPTION: This snippet demonstrates how to create an Agent programmatically in Haystack using Python. It initializes a search component (SerpAPIComponent), a prompt model (PromptModel), and a calculator pipeline. It then adds these tools to the Agent, enabling it to perform tasks that require web searches and calculations. The example shows how to run the agent with a query.
SOURCE: https://github.com/deepset-ai/haystack/blob/main/proposals/text/3925-mrkl-agent.md#_snippet_3

LANGUAGE: python
CODE:
```
search = SerpAPIComponent(api_key=os.environ.get("SERPAPI_API_KEY"), name="Serp", inputs=["Query"])

prompt_model=PromptModel(model_name_or_path="text-davinci-003", api_key=os.environ.get("OPENAI_API_KEY"))

calculator = Pipeline()
calculator.add_node(PromptNode(
    model_name_or_path=prompt_model,
    default_prompt_template=PromptTemplate(prompt_text="Write a simple python function that calculates..."),
    output_variable="python_runtime_input")  # input
calculator.add_node(PythonRuntime())  # actual calculator

prompt_node = PromptNode(
    model_name_or_path=prompt_model,
    stop_words=["Observation:"]
)

agent = Agent(prompt_node=prompt_node)
# Nodes and pipelines can be added as tools to the agent. Just as nodes can be added to pipelines with add_node()
agent.add_tool("Search", search, "useful for when you need to answer questions about current events. You should ask targeted questions")
agent.add_tool("Calculator", calculator, "useful for when you need to answer questions about math")

result = agent.run("What is 2 to the power of 3?")
```

----------------------------------------

TITLE: Installing Haystack via pip
DESCRIPTION: This code snippet demonstrates how to install the Haystack library using pip, the Python package installer. It installs the core haystack package, enabling users to leverage Haystack's features for question answering and search.
SOURCE: https://github.com/deepset-ai/haystack/blob/main/test/test_files/markdown/sample.md#_snippet_0

LANGUAGE: bash
CODE:
```
pip install farm-haystack
```

----------------------------------------

TITLE: Haystack Pipeline with OpenAI Embedders in Python
DESCRIPTION: This example demonstrates how to set up indexing and query pipelines using Haystack components and OpenAI embedders. It shows the configuration of TxtConverter, PreProcessor, DocumentWriter, OpenAITextEmbedder, OpenAIDocumentEmbedder, MemoryRetriever and Reader, connecting components within the pipelines.
SOURCE: https://github.com/deepset-ai/haystack/blob/main/proposals/text/5390-embedders.md#_snippet_0

LANGUAGE: python
CODE:
```
from haystack import Pipeline
from haystack.components import (
    TxtConverter,
    PreProcessor,
    DocumentWriter,
    OpenAITextEmbedder,
    OpenAIDocumentEmbedder,
    MemoryRetriever,
    Reader,
)
from haystack.document_stores import MemoryDocumentStore
docstore = MemoryDocumentStore()

indexing_pipe = Pipeline()
indexing_pipe.add_store("document_store", docstore)
indexing_pipe.add_node("txt_converter", TxtConverter())
indexing_pipe.add_node("preprocessor", PreProcessor())
indexing_pipe.add_node("embedder", OpenAIDocumentEmbedder(model_name="text-embedding-ada-002"))
indexing_pipe.add_node("writer", DocumentWriter(store="document_store"))
indexing_pipe.connect("txt_converter", "preprocessor")
indexing_pipe.connect("preprocessor", "embedder")
indexing_pipe.connect("embedder", "writer")

indexing_pipe.run(...)

query_pipe = Pipeline()
query_pipe.add_store("document_store", docstore)
query_pipe.add_node("embedder", OpenAITextEmbedder(model_name="text-embedding-ada-002"))
query_pipe.add_node("retriever", MemoryRetriever(store="document_store", retrieval_method="embedding"))
query_pipe.add_node("reader", Reader(model_name="deepset/model-name"))
query_pipe.connect("embedder", "retriever")
query_pipe.connect("retriever", "reader")

results = query_pipe.run(...)
```

----------------------------------------

TITLE: Building Indexing and Query Pipelines in Haystack
DESCRIPTION: This code snippet demonstrates how to construct indexing and query pipelines in Haystack 2.0. It showcases the usage of new nodes like `DocumentEmbedder`, `StringEmbedder`, and `DocumentWriter` to decouple retrieval from document stores. This approach avoids tight coupling between `DocumentStore`s and `Retriever`s, allowing for more flexible and maintainable pipelines.
SOURCE: https://github.com/deepset-ai/haystack/blob/main/proposals/text/4370-documentstores-and-retrievers.md#_snippet_0

LANGUAGE: python
CODE:
```
from haystack import Pipeline
from haystack.nodes import (
    TxtConverter,
    PreProcessor,
    DocumentWriter,
    DocumentEmbedder,
    StringEmbedder,
    MemoryRetriever,
    Reader,
)
from haystack.document_stores import MemoryDocumentStore

docstore = MemoryDocumentStore()

indexing_pipe = Pipeline()
indexing_pipe.add_store("document_store", docstore)
indexing_pipe.add_node("txt_converter", TxtConverter())
indexing_pipe.add_node("preprocessor", PreProcessor())
indexing_pipe.add_node("embedder", DocumentEmbedder(model_name="deepset/model-name"))
indexing_pipe.add_node("writer", DocumentWriter(store="document_store"))
indexing_pipe.connect("txt_converter", "preprocessor")
indexing_pipe.connect("preprocessor", "embedder")
indexing_pipe.connect("embedder", "writer")

indexing_pipe.run(...)

query_pipe = Pipeline()
query_pipe.add_store("document_store", docstore)
query_pipe.add_node("embedder", StringEmbedder(model_name="deepset/model-name"))
query_pipe.add_node("retriever", MemoryRetriever(store="document_store", retrieval_method="embedding"))
query_pipe.add_node("reader", Reader(model_name="deepset/model-name"))
query_pipe.connect("embedder", "retriever")
query_pipe.connect("retriever", "reader")

results = query_pipe.run(...)
```

----------------------------------------

TITLE: Running a Query on the QA Pipeline in Python
DESCRIPTION: This code snippet demonstrates how to run a query on the previously defined QA pipeline. The pipeline processes the query and returns the answer generated by the PromptNode, leveraging the retriever for context.
SOURCE: https://github.com/deepset-ai/haystack/blob/main/proposals/text/4172-shaper-in-prompt-template.md#_snippet_1

LANGUAGE: python
CODE:
```
p.run(
    query="What is the most popular drink?"
)
```

----------------------------------------

TITLE: Basic Pipeline Evaluation Example in Haystack
DESCRIPTION: This snippet demonstrates a basic evaluation of a Haystack pipeline using the `eval` function. It takes a pipeline, input data, and expected output data as input, calculates metrics using Semantic Answer Similarity (SAS), and saves the results to a CSV file. This provides a high-level overview of how to evaluate pipelines in Haystack 2.0.
SOURCE: https://github.com/deepset-ai/haystack/blob/main/proposals/text/5794-evaluation-haystack-2.md#_snippet_0

LANGUAGE: python
CODE:
```
pipe = Pipeline()
...
inputs = [{"component_name": {"query": "some question"}, ...}, ...]
expected_output = [{"another_component_name": {"answer": "42"}, ...}, ...]
result = eval(pipe, inputs=inputs, expected_output=expected_output)
metrics = result.calculate_metrics(Metric.SAS)
metrics.save("path/to/file.csv")
```

----------------------------------------

TITLE: Creating and Running a Pipeline in Python
DESCRIPTION: This code creates a Haystack pipeline, adds `AddValue` and `Double` nodes to it, connects them, and runs the pipeline with an initial value of 1. Parameters are passed during the `add_node` and `run` stages. The final result is asserted to be 18.
SOURCE: https://github.com/deepset-ai/haystack/blob/main/proposals/text/4284-drop-basecomponent.md#_snippet_2

LANGUAGE: python
CODE:
```
pipeline = Pipeline()

# Nodes can be initialized as standalone objects.
# These instances can be added to the Pipeline in several places.
addition = AddValue(add=1)

# Nodes are added with a name and an node. Note the lack of references to any other node.
pipeline.add_node("first_addition", addition, parameters={"add": 3})  # Nodes can store default parameters per node.
pipeline.add_node("second_addition", addition)  # Note that instances can be reused
pipeline.add_node("double", Double())

# Nodes are the connected as input node: [list of output nodes]
pipeline.connect(connect_from="first_addition", connect_to="double")
pipeline.connect(connect_from="double", connect_to="second_addition")

pipeline.draw("pipeline.png")

# Pipeline.run() accepts 'data' and 'parameters' only. Such dictionaries can contain
# anything, depending on what the first node(s) of the pipeline requires.
# Pipeline does not validate the input: the first node(s) should do so.
results = pipeline.run(
    data={"value": 1},
    parameters = {"second_addition": {"add": 10}}   # Parameters can be passed at this stage as well
)
assert results == {"value": 18}
```

----------------------------------------

TITLE: TableReader Basic Usage with Pandas DataFrame
DESCRIPTION: This example demonstrates how to use the TableReader with a pandas DataFrame and shows how to access the answer context using row and column indices after the introduction of the TableCell dataclass. It showcases a basic table question answering pipeline using Haystack's TableReader.
SOURCE: https://github.com/deepset-ai/haystack/blob/main/proposals/text/3875-table-cell.md#_snippet_0

LANGUAGE: python
CODE:
```
import pandas as pd
from haystack.nodes import TableReader
from haystack import Document

data = {
    "actors": ["brad pitt", "leonardo di caprio", "george clooney"],
    "age": ["58", "47", "60"],
    "number of movies": ["87", "53", "69"],
    "date of birth": ["18 december 1963", "11 november 1974", "6 may 1961"],
}
table_doc = Document(content=pd.DataFrame(data), content_type="table")
reader = TableReader(model_name_or_path="google/tapas-base-finetuned-wtq", max_seq_len=128)
prediction = reader.predict(query="Who was in the most number of movies?", documents=[table_doc])
answer = prediction["answers"][0]

# New feature
# answer.context -> [["actor", "age", "number of movies"], ["Brad Pitt",...], [...]]
# answer.offsets_in_context[0] -> (row=1, col=1)
print(answer.context[answer.offsets_in_context[0].row][answer.offsets_in_context[0].col])
```

----------------------------------------

TITLE: Defining FAQ Indexing Pipeline in YAML - Haystack
DESCRIPTION: This YAML configuration defines an FAQ indexing pipeline in Haystack. It includes components like ElasticsearchDocumentStore, EmbeddingRetriever, and CsvTextConverter. The indexing pipeline processes CSV files containing question-answer pairs, retrieves embeddings for the questions, and stores the documents in the document store. It leverages sentence-transformers/all-MiniLM-L6-v2 for embeddings and requires Elasticsearch to be running locally.
SOURCE: https://github.com/deepset-ai/haystack/blob/main/proposals/text/3550-csv-converter.md#_snippet_0

LANGUAGE: yaml
CODE:
```
# To allow your IDE to autocomplete and validate your YAML pipelines, name them as <name of your choice>.haystack-pipeline.yml

version: ignore

components:    # define all the building-blocks for Pipeline
  - name: DocumentStore
    type: ElasticsearchDocumentStore
    params:
      host: localhost
      embedding_field: question_emb
      embedding_dim: 384
      excluded_meta_data:
        - question_emb
      similarity: cosine
  - name: Retriever
    type: EmbeddingRetriever
    params:
      document_store: DocumentStore    # params can reference other components defined in the YAML
      embedding_model: sentence-transformers/all-MiniLM-L6-v2
      scale_score: False
  - name: CSVConverter
    type: CsvTextConverter

pipelines:
  - name: indexing
    nodes:
      - name: CSVConverter
        inputs: [File]
      - name: Retriever
        inputs: [ CSVConverter ]
      - name: DocumentStore
        inputs: [ Retriever ]
```

----------------------------------------

TITLE: Initializing PromptNode
DESCRIPTION: This code snippet demonstrates how to instantiate a PromptNode with a specified LLM model and use it to perform a question-answering task. It shows the basic usage of PromptNode with a natural language prompt.
SOURCE: https://github.com/deepset-ai/haystack/blob/main/proposals/text/3665-prompt-node.md#_snippet_0

LANGUAGE: python
CODE:
```
	  from haystack.nodes.llm import PromptNode
      pn = PromptNode(model_name_or_path="google/flan-t5-base")
      pn("What is the capital of Germany?")
```

----------------------------------------

TITLE: Initializing a Generative QA Pipeline with PromptNode in Python
DESCRIPTION: This code initializes a generative QA pipeline using Haystack's PromptNode. It sets up an InMemoryDocumentStore, an EmbeddingRetriever, and a PromptNode with a default prompt template. The components are then connected in a pipeline to enable question-answering with references.
SOURCE: https://github.com/deepset-ai/haystack/blob/main/proposals/text/4172-shaper-in-prompt-template.md#_snippet_0

LANGUAGE: python
CODE:
```
from haystack import Pipeline
from haystack.document_store import InMemoryDocumentStore
from haystack.nodes import PromptNode, EmbeddingRetriever

document_store = InMemoryDocumentStore()
retriever = EmbeddingRetriever(document_store=document_store, ...)
pn = PromptNode(default_prompt_template="question-answering-with-references")

p = Pipeline()
p.add_node(component=retriever, name="Retriever", inputs=["Query"])
p.add_node(component=pn, name="Prompt", inputs=["Retriever"])
```

----------------------------------------

TITLE: RAG Pipeline Definition with Haystack 2.0 Components (Python)
DESCRIPTION: This code defines a Retrieval-Augmented Generation (RAG) pipeline using Haystack 2.0 components. It showcases the use of MemoryRetriever, PromptBuilder, ChatGPTGenerator, and RepliesToAnswersConverter components within a Pipeline object. The pipeline is configured to connect these components and process questions to generate answers.
SOURCE: https://github.com/deepset-ai/haystack/blob/main/proposals/text/5540-llm-support-2.0.md#_snippet_0

LANGUAGE: python
CODE:
```
from haystack.preview.components import MemoryRetriever, PromptBuilder, ChatGPTGenerator, RepliesToAnswersConverter
from haystack.preview.document_stores import MemoryDocumentStore
from haystack.preview.pipeline import Pipeline

pipe = Pipeline()
pipe.add_store("store", MemoryDocumentStore())
pipe.add_component("retriever", MemoryRetriever(), store="store")
pipe.add_component("prompt_builder", PromptBuilder("deepset/question-answering"))
pipe.add_component("llm", GPT4Generator(api_key="..."))
pipe.add_component("replies_converter", RepliesToAnswersConverter())

pipe.connect("retriever", "prompt_builder")
pipe.connect("prompt_builder", "llm")
pipe.connect("llm", "replies_converter")

questions = ["Why?", "Why not?"]
results = pipe.run({
	"retriever": {"queries": questions},
	"prompt_builder": {"questions": questions},
})

assert results == {
	"replies_converter": {
    "answers": [[Answer("Because of this.")], [Answer("Because of that.")]]
  }
}
```

----------------------------------------

TITLE: PromptNode YAML Configuration
DESCRIPTION: This YAML configuration defines a Haystack pipeline with two PromptNode components, a PromptModel, a PromptTemplate, and a retriever. It showcases how to configure the pipeline declaratively, reuse the PromptModel instance, and define a custom prompt template.
SOURCE: https://github.com/deepset-ai/haystack/blob/main/proposals/text/3665-prompt-node.md#_snippet_13

LANGUAGE: yaml
CODE:
```
components:

  # can go in pipeline
  - name: prompt_node
    params:
      prompt_template: template
      model_name_or_path: model
      output_variable: "questions"
    type: PromptNode

  # can go in pipeline
  - name: prompt_node_2
    params:
      prompt_template: "question-answering"
      model_name_or_path: deepset/model-name
    type: PromptNode

  # not in pipeline - only needed if you're reusing the model across multiple PromptNode in a pipeline
  # and hidden from users in the Python beginner world
  - name: model
    params:
      model_name_or_path: google/flan-t5-xl
    type: PromptModel

  # not in pipeline
  - name: template
    params:
      name: "question-generation-v2"
      prompt_text: "Given the following $documents, please generate a question. Question:"
      input_variables: documents
    type: PromptTemplate

pipelines:
  - name: question-generation-answering-pipeline
    nodes:
      - name: EmbeddingRetriever
        inputs: [Query]
      - name: prompt_node
        inputs: [EmbeddingRetriever]
      - name: prompt_node_2
        inputs: [prompt_node]
```

----------------------------------------

TITLE: LLM Generator Component Interface (Python)
DESCRIPTION: This code defines the interface for an LLM generator component, specifically ChatGPTGenerator, in Haystack 2.0. It inherits from a generic component class and defines a `run` method that takes a list of prompts as input and returns a dictionary containing a list of replies. The component is designed to be modular and reusable with other components in a pipeline.
SOURCE: https://github.com/deepset-ai/haystack/blob/main/proposals/text/5540-llm-support-2.0.md#_snippet_1

LANGUAGE: python
CODE:
```
@component
class ChatGPTGenerator:

    @component.output_types(replies=List[List[str]])
    def run(self, prompts: List[str], ... chatgpt specific params...):
        ...
        return {'replies': [...]}
```

----------------------------------------

TITLE: Pipeline Configuration JSON
DESCRIPTION: This JSON configuration file demonstrates the structure for defining dependencies, stores, nodes, and pipelines. It includes configurations for sparse and dense retrieval, ranking, and reading components.
SOURCE: https://github.com/deepset-ai/haystack/blob/main/proposals/text/4284-drop-basecomponent.md#_snippet_5

LANGUAGE: json
CODE:
```
{
    # A list of "dependencies" for the application.
    # Used to ensure all external nodes are present when loading.
    "dependencies" : [
        "haystack == 2.0.0",
        "my_custom_node_module == 0.0.1",
    ],

    # Stores are defined here, outside single pipeline graphs.
    # All pipelines have access to all these docstores.
    "stores": {
        # Nodes will be able to access them by the name defined here,
        # in this case `my_first_store` (see the retrievers below).
        "my_first_store": {
            # class_name is mandatory
            "class_name": "InMemoryDocumentStore",
            # Then come all the additional parameters for the store
            "use_bm25": true
        },
        "my_second_store": {
            "class_name": "InMemoryDocumentStore",
            "use_bm25": false
        }
    },

    # Nodes are defined here, outside single pipeline graphs as well.
    # All pipelines can use these nodes. Instances are re-used across
    # Pipelines if they happen to share a node.
    "nodes": {
        # In order to reuse an instance across multiple nodes, instead
        # of a `class_name` there should be a pointer to another node.
        "my_sparse_retriever": {
            # class_name is mandatory, unless it's a pointer to another node.
            "class_name": "BM25Retriever",
            # Then come all the additional init parameters for the node
            "store_name": "my_first_store",
            "top_k": 5
        },
        "my_dense_retriever": {
            "class_name": "EmbeddingRetriever",
            "model_name": "deepset-ai/a-model-name",
            "store_name": "my_second_store",
            "top_k": 5
        },
        "my_ranker": {
            "class_name": "Ranker",
            "inputs": ["documents", "documents"],
            "outputs": ["documents"],
        },
        "my_reader": {
            "class_name": "Reader",
            "model_name": "deepset-ai/another-model-name",
            "top_k": 3
        }
    },

    # Pipelines are defined here. They can reference all nodes above.
    # All pipelines will get access to all docstores
    "pipelines": {
        "sparse_question_answering": {
            # Mandatory list of edges. Same syntax as for `Pipeline.connect()`
            "edges": [
                ("my_sparse_retriever", ["reader"])
            ],
            # To pass some parameters at the `Pipeline.add_node()` stage, add them here.
            "parameters": {
                "my_sparse_retriever": {
                    "top_k": 10
                }
            },
            # Metadata can be very valuable for dC and to organize larger Applications
            "metadata": {
                "type": "question_answering",
                "description": "A test pipeline to evaluate Sparse QA.",
                "author": "ZanSara"
            },
            # Other `Pipeline.__init__()` parameters
            "max_allowed_loops": 10,
        },
        "dense_question_answering": {
            "edges": [
                ("my_dense_retriever", ["reader"])
            ],
            "metadata": {
                "type": "question_answering",
                "description": "A test pipeline to evaluate Sparse QA.",
                "author": "an_intern"
            }
        },
        "hybrid_question_answering": {
            "edges": [
                ("my_sparse_retriever", ["ranker"]),
                ("my_dense_retriever", ["ranker"]),
                ("ranker", ["reader"]),
            ],
            "metadata": {
                "type": "question_answering",
                "description": "A test pipeline to evaluate Hybrid QA.",
                "author": "the_boss"
            }
        }
    }
}
```

----------------------------------------

TITLE: Initializing PromptNode with Template
DESCRIPTION: This code demonstrates how to initialize the PromptNode with a specified model and a default prompt template name. Subsequently the PromptNode is called directly, which will run the specified template on the provided documents. This showcases the usage of a predefined template during initialization.
SOURCE: https://github.com/deepset-ai/haystack/blob/main/proposals/text/3665-prompt-node.md#_snippet_10

LANGUAGE: python
CODE:
```
      from haystack.nodes.llm import PromptNode
      pn = PromptNode(model_name_or_path="google/flan-t5-base", prompt_template="question-generation")
      pn(documents=["Berlin is the capital of Germany."])
```

----------------------------------------

TITLE: PromptNode Pipeline Example
DESCRIPTION: This code demonstrates how to use PromptNode within a Haystack pipeline to retrieve documents and then answer a question using the retrieved documents as context. It showcases the integration of EmbeddingRetriever and PromptNode.
SOURCE: https://github.com/deepset-ai/haystack/blob/main/proposals/text/3665-prompt-node.md#_snippet_11

LANGUAGE: python
CODE:
```
from haystack.pipelines import PromptNode
top_k = 3
query = "Who are the parents of Arya Stark?"
retriever = EmbeddingRetriever(...)
pn = PromptNode(model_name_or_path="google/flan-t5-base", prompt_template="question-answering")

pipe = Pipeline()
pipe.add_node(component=retriever, name="Retriever", inputs=["Query"])
pipe.add_node(component=pn, name="prompt_node", inputs=["Retriever"])

output = pipe.run(query=query,
                  params={"Retriever": {"top_k": top_k}},
                  questions=[query for n in range(0, top_k)],
                  #documents parameter we need for this task will be automatically populated by the retriever
                  )

output["results"]
```

----------------------------------------

TITLE: Pipeline with MetaFieldRanker in Haystack (Python)
DESCRIPTION: This snippet illustrates how to integrate the MetaFieldRanker into a Haystack pipeline. It showcases the addition of an InMemoryBM25Retriever and the MetaFieldRanker, connecting them to enable document retrieval followed by ranking based on the 'rating' meta field.  The retriever fetches documents and passes them to the ranker for sorting.
SOURCE: https://github.com/deepset-ai/haystack/blob/main/proposals/text/6141-meta-field-ranker.md#_snippet_1

LANGUAGE: python
CODE:
```
pipeline = Pipeline()
pipeline.add_component(component=InMemoryBM25Retriever(document_store=document_store, top_k=20)
, name="Retriever")
pipeline.add_component(component=MetaFieldRanker(meta_field="rating"), name="Ranker")
pipeline.connect("Retriever.documents", "MetaFieldRanker.documents")
```

----------------------------------------

TITLE: Using PromptNode with Template Name
DESCRIPTION: This code snippet illustrates how to use the `prompt` method of PromptNode with a specified prompt template name to perform a task. The example shows question generation using the documents parameter.
SOURCE: https://github.com/deepset-ai/haystack/blob/main/proposals/text/3665-prompt-node.md#_snippet_3

LANGUAGE: python
CODE:
```
      from haystack.nodes.llm import PromptNode
      pn = PromptNode(model_name_or_path="google/flan-t5-base")
	  pn.prompt("question-generation", documents=["Berlin is the capital of Germany."])
```

----------------------------------------

TITLE: Selecting Default Prompt Template
DESCRIPTION: This code snippet shows how to select a specific default template for a task using `use_prompt_template` and then subsequently use it. It generates a question using the specified template and provided documents.
SOURCE: https://github.com/deepset-ai/haystack/blob/main/proposals/text/3665-prompt-node.md#_snippet_4

LANGUAGE: python
CODE:
```
	  qa = pn.use_prompt_template("deepset/question-generation-v2")
      qa(documents=["Berlin is the capital of Germany."])
```

----------------------------------------

TITLE: Agent Configuration in YAML Format
DESCRIPTION: This snippet shows an example of an Agent configuration defined in a YAML file. It specifies the components, including PromptNodes, PromptModels, SerpAPIComponent, and PythonRuntime, along with their parameters. It also defines a calculator pipeline and an agent, specifying the tools and their descriptions. The YAML configuration allows for easy and declarative agent setup.
SOURCE: https://github.com/deepset-ai/haystack/blob/main/proposals/text/3925-mrkl-agent.md#_snippet_4

LANGUAGE: yaml
CODE:
```
version: ignore

components:
  - name: AgentPromptNode
    type: PromptNode
    params:
      model_name_or_path: DavinciModel
      stop_words: ['Observation:']
  - name: DavinciModel
    type: PromptModel
    params:
      model_name_or_path: 'text-davinci-003'
      api_key: 'XYZ'
  - name: Serp
    type: SerpAPIComponent
    params:
      api_key: 'XYZ'
  - name: CalculatorInput
    type: PromptNode
    params:
      model_name_or_path: DavinciModel
      default_prompt_template: CalculatorTemplate
      output_variable: python_runtime_input
  - name: Calculator
    type: PythonRuntime
  - name: CalculatorTemplate
    type: PromptTemplate
    params:
      name: calculator
      prompt_text:  |
          # Write a simple python function that calculates
          # $query
          # Do not print the result; invoke the function and assign the result to final_result variable
          # Start with import statement

pipelines:
  - name: calculator_pipeline
    nodes:
      - name: CalculatorInput
        inputs: [Query]
      - name: Calculator
        inputs: [CalculatorInput]

agents:
  - name: agent
    params:
      prompt_node: AgentPromptNode
      tools:
        - name: Search
          pipeline_or_node: Serp
          description: >
            useful for when you need to answer questions about current events.
            You should ask targeted questions
        - name: Calculator
          pipeline_or_node: calculator_pipeline
          description: >
            useful for when you need to answer questions about math
```

----------------------------------------

TITLE: Multi-PromptNode Pipeline
DESCRIPTION: This code shows how to use multiple PromptNode components in a Haystack pipeline. It generates questions from retrieved documents using one PromptNode and then answers those questions using another PromptNode, demonstrating how to bind the output of one node to the input of another.
SOURCE: https://github.com/deepset-ai/haystack/blob/main/proposals/text/3665-prompt-node.md#_snippet_12

LANGUAGE: python
CODE:
```
top_k = 3
query = "Who are the parents of Arya Stark?"
retriever = EmbeddingRetriever(...)
model = PromptModel(model_name_or_path="google/flan-t5-small")

qg = PromptNode(prompt_template="question-generation", prompt_model=model, output_variable="questions")
qa = PromptNode(prompt_template="question-answering", prompt_model=model)

pipe = Pipeline()
pipe.add_node(component=retriever, name="Retriever", inputs=["Query"])
pipe.add_node(component=qg, name="qg", inputs=["Retriever"])
pipe.add_node(component=qa, name="qa", inputs=["qg"])

result = pipe.run(query=query)

print(result["results"])
```

----------------------------------------

TITLE: Defining HFTextEmbedder and HFDocumentEmbedder Components in Python
DESCRIPTION: This code defines the structure of `HFTextEmbedder` and `HFDocumentEmbedder` components in Python using decorators.  The `HFTextEmbedder` is for embedding strings into vectors and the `HFDocumentEmbedder` embeds `Document` objects.  The `run` methods are shown with their input and output types.  Dependencies: `haystack` framework.
SOURCE: https://github.com/deepset-ai/haystack/blob/main/proposals/text/5390-embedders.md#_snippet_1

LANGUAGE: python
CODE:
```
@component
class HFTextEmbedder:
    ...

    @component.output_types(result=List[np.ndarray])
    def run(self, strings: List[str]):
        ...
        return {"result": list_of_computed_embeddings}


@component
class HFDocumentEmbedder:
    ...

    @component.output_types(result=List[Document])
    def run(self, documents: List[Document]):
        ...
        return {"result": list_of_documents_with_embeddings}
```

----------------------------------------

TITLE: DeepEvalEvaluator Pipeline Integration - Python
DESCRIPTION: This code demonstrates how to integrate the DeepEvalEvaluator component into a Haystack pipeline for evaluating RAG outputs. It shows how to add the component to the pipeline, specify the metric and its parameters, and run the pipeline with questions, contexts, and answers as input. The expected output is a DeepEvalResult object containing the metric and its score.
SOURCE: https://github.com/deepset-ai/haystack/blob/main/proposals/text/6784-integrations-for-eval-framworks.md#_snippet_0

LANGUAGE: python
CODE:
```
p = Pipeline()
p.add_component(instance=DeepEvalEvaluator(metric="Hallucination", params={"threshold": 0.3)}, name="evaluator"))
# p.add_component(instance=RagasEvaluator()...

questions = [...]
contexts = [...]
answers = [...]

p.run({"evaluator": {"questions": questions, "context": contexts, "answer": answers})
# {"evaluator": DeepEvalResult(metric='hallucination', score=0.817)}
```

----------------------------------------

TITLE: SentenceTransformers Embedder Initialization in Python
DESCRIPTION: This example shows how to initialize `SentenceTransformersTextEmbedder` and `SentenceTransformersDocumentEmbedder` with different models for DPR (Dense Passage Retrieval). It is used to encode queries and documents separately. No dependencies are explicitly shown.
SOURCE: https://github.com/deepset-ai/haystack/blob/main/proposals/text/5390-embedders.md#_snippet_2

LANGUAGE: python
CODE:
```
dpr_query_embedder = SentenceTransformersTextEmbedder(model_name="facebook/dpr-question_encoder-single-nq-base")
dpr_doc_embedder = SentenceTransformersDocumentEmbedder(model_name="facebook/dpr-ctx_encoder-single-nq-base")
```

----------------------------------------

TITLE: Conversion from MongoDB-like to New Filter Style (JSON)
DESCRIPTION: This example shows how a filter written in a MongoDB-like style (used in Haystack 1.x) can be converted to the new, more structured filter format proposed for Haystack 2.x. It provides a clear mapping between the old and new representations for complex conditions involving multiple fields and operators.
SOURCE: https://github.com/deepset-ai/haystack/blob/main/proposals/text/6001-document-store-filter-rework.md#_snippet_4

LANGUAGE: json
CODE:
```
{
    "$and": {
        "type": {"$eq": "article"},
        "$or": {"genre": {"$in": ["economy", "politics"]}, "publisher": {"$eq": "nytimes"}},
        "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
        "rating": {"$gte": 3},
    }
}
```

LANGUAGE: json
CODE:
```
{
    "operator": "AND",
    "conditions": [
        { "field": "type", "operator": "==", "value": "article" },
        {
            "operator": "OR",
            "conditions": [
                { "field": "genre", "operator": "in", "value": ["economy", "politics"] },
                { "field": "publisher", "operator": "==", "value": "nytimes" },
            ]
        },
        { "field": "date", "operator": ">=", "value": "2015-01-01" },
        { "field": "date", "operator": "<", "value": "2021-01-01" },
        { "field": "rating", "operator": ">=", "value": 3 }
    ]
}
```

----------------------------------------

TITLE: Initializing MetaFieldRanker in Haystack (Python)
DESCRIPTION: This snippet demonstrates how to instantiate a MetaFieldRanker with specific parameters such as the meta field to sort by (`meta_field`), the weight to assign to the ranking (`weight`), the sorting order (`ascending`), and the number of top documents to return (`top_k`).  It shows a basic example of how to configure the ranker to sort documents based on a 'rating' meta field.
SOURCE: https://github.com/deepset-ai/haystack/blob/main/proposals/text/6141-meta-field-ranker.md#_snippet_0

LANGUAGE: python
CODE:
```
ranker = MetaFieldRanker(
    meta_field="rating",
    weight="0.5",
    ascending=False,
    top_k=3,
)
```

----------------------------------------

TITLE: Full Agent Trace Example
DESCRIPTION: This snippet presents a complete interaction trace of the MRKLAgent, including the initial question, the LLM's 'Thought', chosen 'Action', 'Action Input', and the 'Observation' from the invoked tool (Search or Calculator). It demonstrates how the agent iteratively answers the question by breaking it down into sub-questions and utilizing different tools.
SOURCE: https://github.com/deepset-ai/haystack/blob/main/proposals/text/3925-mrkl-agent.md#_snippet_2

LANGUAGE: text
CODE:
```
Answer the following questions as best as you can. You have access to the following tools:

Search: useful for when you need to answer questions about current events. You should ask targeted questions
Calculator: useful for when you need to answer questions about math

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [Search, Calculator]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
Question: Who is Olivia Wilde's boyfriend? What is his current age raised to the 0.23 power?
Thought: I need to do some research to answer this question.
Action: Search
Action Input: Olivia Wilde's boyfriend
Observation: First linked in November 2011, Wilde and Sudeikis got engaged in January 2013. They later became parents, welcoming son Otis in 2014 and daughter Daisy in 2016.
Thought: I need to find out his age
Action: Search
Action Input: Jason Sudeikis age
Observation: 47 years
Thought: I need to raise it to the 0.23 power
Action: Calculator
Action Input: 47^0.23
Observation: 2.4242784855673896
Thought: I now know the final answer
Final Answer: Jason Sudeikis, Olivia Wilde's boyfriend, is 47 years old and his age raised to the 0.23 power is 2.4242784855673896.
```

----------------------------------------

TITLE: Defining a Haystack Node: AddValue in Python
DESCRIPTION: This code defines a custom Haystack node called `AddValue` which adds a specified value to the input data. It uses the `@node` decorator and has a `run()` method that performs the addition. The node takes an input named 'value' and outputs a value.
SOURCE: https://github.com/deepset-ai/haystack/blob/main/proposals/text/4284-drop-basecomponent.md#_snippet_0

LANGUAGE: python
CODE:
```
from typing import Dict, Any, List, Tuple
from haystack.pipeline import Pipeline
from haystack.nodes import node

# A Haystack Node. See below for details about this contract.
# Crucial components are the @node decorator and the `run()` method
@node
class AddValue:
    def __init__(self, add: int = 1, input_name: str = "value", output_name: str = "value"):
        self.add = add
        self.init_parameters = {"add": add}
        self.inputs = [input_name]
        self.outputs = [output_name]

    def run(
        self,
        name: str,
        data: List[Tuple[str, Any]],
        parameters: Dict[str, Any],
        stores: Dict[str, Any],
    ):
        my_parameters = parameters.get(name, {})
        add = my_parameters.get("add", self.add)

        for _, value in data:
            value += add

        return ({self.outputs[0]: value}, parameters)
```

----------------------------------------

TITLE: Initializing Haystack EmbeddingRetriever with Encoder (Proposed Method) - Python
DESCRIPTION: Shows the proposed method for initializing Haystack's `EmbeddingRetriever` by passing a pre-configured encoder object (`encoder`). This decouples the specific embedding logic from the retriever itself, allowing users to easily plug in different encoders, including custom ones. The retriever then uses the provided encoder for embedding tasks, simplifying the retriever's constructor parameters.
SOURCE: https://github.com/deepset-ai/haystack/blob/main/proposals/text/3558-embedding_retriever.md#_snippet_2

LANGUAGE: python
CODE:
```
retriever = EmbeddingRetriever(
	document_store=document_store,
	encoder=encoder
  )
```

----------------------------------------

TITLE: Installing Docker and Docker Compose
DESCRIPTION: This code snippet provides commands to update the package manager and install Docker and Docker Compose on a Linux system. These are prerequisites for running the Haystack demo application using Docker Compose.
SOURCE: https://github.com/deepset-ai/haystack/blob/main/test/test_files/markdown/sample.md#_snippet_1

LANGUAGE: bash
CODE:
```
    # apt-get update && apt-get install docker && apt-get install docker-compose
    # service docker start
```

----------------------------------------

TITLE: Adding Prompt Template
DESCRIPTION: This code snippet demonstrates how to add a new prompt template using the `add_prompt_template` method. It then lists the available templates using `get_prompt_templates_names` to show the newly added template.
SOURCE: https://github.com/deepset-ai/haystack/blob/main/proposals/text/3665-prompt-node.md#_snippet_5

LANGUAGE: python
CODE:
```
      from haystack.nodes.llm import PromptNode
      PromptNode.add_prompt_template(PromptTemplate(name="sentiment-analysis",
                              prompt_text="Please give a sentiment for this context. Answer with positive, "
                              "negative or neutral. Context: $documents; Answer:",
                              input_variables=["documents"]))
      PromptNode.get_prompt_templates_names()
```