# -----------------------------------------------------------------------
# Mini-Project: Groundwork Coffee Co. Q&A Assistant
# -----------------------------------------------------------------------
""" 
This project implements a Q&A assistant for Groundwork Coffee Co.
It uses a Retrieval-Augmented Generation (RAG) approach to answer questions
based on the company's documents.
"""

from dotenv import load_dotenv
import os
from pathlib import Path

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

# ===========================================
# Step 1: Setup
print("\nStep 1: Setup\n")
# ===========================================

# getting the documents directory path
docs_dir = Path("../../python-200-v1/lessons/06_AI_augmentation/resources/groundwork_docs")
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"

# loading the environment variables from the .env file
if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")


# ===========================================
# Step 2: Load the Documents
print("\nStep 2: Load the Documents\n")
# ===========================================

documents = SimpleDirectoryReader(input_dir=docs_dir).load_data()
print(f"Successfully loaded {len(documents)} documents:")
for doc in documents:
    print(f"- {doc.metadata['file_name']}")



# ===========================================
# Step 3: Build the Index and Query Engine
print("\nStep 3: Build the Index and Query Engine\n")
# ===========================================


# setting the embedding model for LlamaIndex
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# setting the LLM model for LlamaIndex
Settings.llm = OpenAI(model="gpt-4o-mini")  

# Build a vector index automatically (handles chunking + embeddings)
index = VectorStoreIndex.from_documents(documents)

# create a query engine from the built index
query_engine = index.as_query_engine(similarity_top_k=3)



print("Index built successfully. Ready to answer questions.")


# ===========================================
# Step 4: Query the Assistant
print("\nStep 4: Query the Assistant\n")
# ===========================================

questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for q in questions:
    print("\n")
    print("=" * 30)
    print(f"Q: {q}")
    print("=" * 30)
    response = query_engine.query(q)
    print("A:", response)
    print("=" * 30)

    for node_with_score in response.source_nodes:
        print(f"File name: {node_with_score.node.metadata['file_name']}")
        print("-" * 30)
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print("-" * 30)
        print(f"Text Snippet: {node_with_score.node.get_content()[:200]}...")
        print("-" * 30)

# Reflect on the assistant's responses
# Did the assistant sound confident and accurate? Did any of the answers surprise you?
# - All answers are short, confident and concise.
# - I'm surprized that for the question about weekend hours the actual relevant documents is just third in the list of retrieved nodes.



# ===========================================
# Step 5: Find a Failure
print("\nStep 5: Find a Failure\n")
# ===========================================

struggle_questions = [
    "Will you be open on New Year?",
    "What is your favorite drink?",
    "Tell me the coffee history from the very beginning."
]

for q in struggle_questions:
    print("\n")
    print("=" * 30)
    print(f"Q: {q}")
    print("=" * 30)
    response = query_engine.query(q)
    print("A:", response)
    print("=" * 30)

    for node_with_score in response.source_nodes:
        print(f"File name: {node_with_score.node.metadata['file_name']}")
        print("-" * 30)
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print("-" * 30)
        print(f"Text Snippet: {node_with_score.node.get_content()[:200]}...")
        print("-" * 30)

# Reflection on struggle questions
# - Assistant tried to uncertainly guess that the shop would be open on New Year, it correctly provides that just Thanksgiving and Christmas are the exceptions.
# - For the question about the favorite drink, the assistant admits it doesn't have a personal preference, which is appropriate. 
# Then he confidently provides the info about drinks availble which is great.
# - The question regarding the history of coffee the assistant is handling great, since he certantly said that he hasn`t that info 
# but he has a history of the coffee shop.
# - Overall, the assistant demonstrates a good balance between acknowledging its limitations and providing helpful information.
# - I would just suggest not to guess about open or not. It would be better just to provide official info or suggest checking with the shop for confirmation.



# ===========================================
# Step 6: Reflection
print("\nStep 6: Reflection\n")
# ===========================================

print("Reflection is given in a comment block.")

"""The LlamaIndex lesson lists the steps that LlamaIndex automates: chunking, embedding, and indexing. 
How many lines of code did your LlamaIndex implementation take in your project? 

- My LlamaIndex implementation took around 30 lines of code without comments. 

Compare that with the keyword RAG code from the Keyword-based RAG lesson, 
or with the code you would need to write each of those steps yourself. 

- The keyword RAG code from the lesson took significantly more lines, often over 100, to achieve the same functionality.

What does that tell you about the value of using a framework?

- Using a framework like LlamaIndex significantly reduces the amount of code needed to implement complex tasks, 
making development more efficient and less error-prone.
"""