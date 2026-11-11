from dotenv import load_dotenv
import os
import string



if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")


# ------------------------------------------------------------------------------------
# --- RAG Concepts --- 
print("\n ---  RAG Concepts ---\n")
# ------------------------------------------------------------------------------------


""" Concepts Question 1 """
print("\t Concepts Question 1\n")

""" Scenario A: A legal team wants an assistant that can answer questions about their internal policy library — 
hundreds of PDFs that are updated every quarter."""
# RAG would be suitable for this scenario.
# - The knowledge base consists of hundreds of documents that change frequently (quarterly).
# - RAG allows dynamic retrieval of the most up-to-date policy chunks without the need for expensive, 
# - time-consuming model retraining or exceeding prompt token limits.

"""Scenario B: A startup wants their model to write product copy in a very specific brand voice — 
a dry, minimalist style that does not appear much online. 
They have 3,000 examples their in-house writers produced over the years."""
# Fine-Tuning would be a perfect fit.
# - The goal is learning a specific stylistic nuance, tone, and brand voice rather than factual retrieval. 
# - Having 3,000 consistent examples provides sufficient training data 
# - to adapt the model's behavioral weights to match that exact writing style.

"""Scenario C: A data analyst needs to ask an LLM questions about a single two-page report she just received. 
She does not need this to work for any other document."""
# Just Prompt Engineering would be enough.
# - The task is a one-off query over a very short document (two pages). 
# - The text fits comfortably within the LLM's context window, making direct copy-paste or prompt injection 
# - the simplest, fastest, and most cost-effective method without needing a vector index or training.

""" Concepts Question 2 """
print("\t Concepts Question 2\n")

""" Why is a confidently wrong answer more harmful than one that says "I am not sure"? 
Give one example of a real situation where a confident hallucination could cause harm."""
# - A confidently wrong answer can lead users to make decisions based on incorrect information, potentially causing harm.
# - As examples we can find medical questions. Nowdayas lots of users want to get quick answers from AI assistants, 
# - and if the assistant confidently provides incorrect medical advice, it could lead to serious health consequences.
""" Why does the way the model expresses an answer affect how much we trust it? """
# - The way the model expresses an answer can influence user trust. For example, 
# - a model that provides a confident but incorrect answer may be trusted more than one that admits uncertainty, 
# - even if the latter is more accurate.

""" Concepts Question 3 """
print("\t Concepts Question 3\n")

""" 
steps = [
    "Generate a response from the LLM",
    "Extract text from source documents",
    "Receive the user's query",
    "Retrieve the most relevant chunks",
    "Convert text chunks into embeddings",
    "Inject retrieved chunks into the prompt",
    "Split text into chunks",
    "Embed the user's query",
]
"""

# Correct order of steps:

# 1. "Extract text from source documents"
# The system parses and extracts plain, readable text from raw files like PDFs, Word documents, or text files.

# 2. "Split text into chunks"
# The extracted text is divided into smaller, discrete chunks to isolate specific topics and fit token boundaries.

# 3. "Convert text chunks into embeddings"
# An embedding model converts each text chunk into a high-dimensional vector representing its semantic meaning and stores it in an index.

# 4. "Receive the user's query"
# The application receives the user's natural language question or request.

# 5. "Embed the user's query"
# The user's query is converted into a vector embedding using the same embedding model used for the documents.

# 6. "Retrieve the most relevant chunks"
# The system performs a similarity search (e.g., cosine similarity) to find document chunk embeddings closest to the query embedding.

# 7. "Inject retrieved chunks into the prompt"
# The top-matching text chunks are formatted alongside instructions and the user query to serve as factual context.

# 8. "Generate a response from the LLM"
# The language model reads the context-augmented prompt and generates a grounded, accurate response.


# ------------------------------------------------------------------------------------
# --- Keyword RAG --- 
print("\n ---  Keyword RAG ---\n")
# ------------------------------------------------------------------------------------


def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]

""" Keyword Question 1 """
print("\t Keyword Question 1\n")

query = "What are your hours on weekends?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}
query = "How do I sign up for rewards?"


results = simple_keyword_retrieval(query, documents)

# which document was selected and why?
# - Selected best match: loyalty.txt
# - 3 of 4 documents have overlap by 1 token.
# - the chosen document was the last one with overlap

""" Keyword Question 2 """
print("\t Keyword Question 2\n")

query = "Do you have anything without caffeine?"
results = simple_keyword_retrieval(query, documents)

# Which document was selected?
# - No overlapping keywords found.
# Whether keyword RAG got this right — and why or why not
# - No, keyword RAG did not get this right because none of the documents contain the exact keyword "caffeine".
# What kind of retrieval would do better here
# - Semantic retrieval would be more effective because it can understand the meaning of "without caffeine" 
# - and match it to documents that discuss caffeine-free options, even if the exact keyword is not present.

""" Keyword Question 3 """
print("\t Keyword Question 3\n")

query = "How do I sign up for rewards?"

# Write your prediction and your reasoning as a comment first, then run the code to check.
# - No document will be chosen.
# - the only overlap is with "for" with loyalty.txt but "for" is a stopwords.
results = simple_keyword_retrieval(query, documents)

# Was your prediction correct?
# - Yes, no document was chosen as I predicted.

# ------------------------------------------------------------------------------------
# --- Semantic RAG Concepts --- 
print("\n ---  Semantic RAG Concepts ---\n")
# ------------------------------------------------------------------------------------


""" Semantic Question 1 """
print("\t Semantic Question 1\n")

# What is a vector embedding? (1-2 sentences)
# - A vector embedding is a numerical representation of text that captures its semantic meaning. 
# - It allows similar texts to be close together in the vector space, even if they don't share exact words.

# Two text chunks have cosine similarity scores of 0.85 and 0.30 with a given query. 
# Which chunk is more relevant, and what does that number tell you about the relationship between the texts?
# - The chunk with 0.85 is more relevant.
# - Range of cosine similarity is from -1 to 1, where 1 indicates perfect similarity and -1 indicates perfect dissimilarity.

# Why can semantic search find a relevant chunk even when none of the exact words from the query appear in the chunk?
# - Semantic search relies on the meaning of the text rather than exact word matches.
# - Vector embeddings capture semantic similarity, so chunks with similar meanings will be close in the vector space, 
# - even if they don't share exact words.

""" Semantic Question 2 """
print("\t Semantic Question 2\n")

"""
| Feature                    | Keyword RAG                       | Semantic RAG                 |
|----------------------------|-----------------------------------|------------------------------|
| What is compared?          | Exact word overlap                | Semantic similarity          |
| What is retrieved?         | Full document                     | Chunks with similar meaning  |
| Can it handle synonyms?    | No                                | Yes                          |
| Storage format             | Plain text dictionary             | Vector embeddings            |
| Relevance score            | Number of overlapping keywords    | Cosine similarity            |
"""

# ------------------------------------------------------------------------------------
# --- LlamaIndex --- 
print("\n ---  LlamaIndex ---\n")
# ------------------------------------------------------------------------------------

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

import logging
# Get rid of WARNING messages from pypdf
logging.getLogger("pypdf").setLevel(logging.ERROR)

# setting the embedding model for LlamaIndex
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# setting the LLM model for LlamaIndex
Settings.llm = OpenAI(model="gpt-4o-mini")  

""" LlamaIndex Question 1 """
print("\t LlamaIndex Question 1\n")

# Load documents directly from PDFs in the folder
docs = SimpleDirectoryReader("../../python-200-v1/lessons/06_AI_augmentation/resources/brightleaf_pdfs").load_data()

# Build a vector index automatically (handles chunking + embeddings)
index = VectorStoreIndex.from_documents(docs)

print(type(index._vector_store).__name__)

query_engine = index.as_query_engine(similarity_top_k=3)

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]
for q in questions:
    print("\n")
    print("=" * 30)
    print(f"Q: {q}")
    print("=" * 30)
    response = query_engine.query(q)
    print("A:", response)
    print("-" * 30)

    for node_with_score in response.source_nodes:
        print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
        print("-" * 30)

# comment on quality of the results
"""What employee benefits does BrightLeaf offer?"""
# Do the retrieved chunks look relevant to the question?
# - Yes, the retrieved chunks look relevant to the question.
# Does the model's response sound confident and specific, or does it hedge with phrases like "based on the context" or "I'm not sure"? Note what you observe about the tone.
# - The model's response sounds confident and specific, providing detailed information about BrightLeaf's employee benefits.
#  Did anything unexpected get retrieved?
# - No, the retrieved chunks are all relevant to the question.

"""What are BrightLeaf's security policies?"""
# Do the retrieved chunks look relevant to the question?
# - Yes, the retrieved chunks look relevant to the question.
# Does the model's response sound confident and specific, or does it hedge with phrases like "based on the context" or "I'm not sure"? Note what you observe about the tone.
# - The model's response sounds confident and specific, providing detailed information about BrightLeaf's security policies.
#  Did anything unexpected get retrieved?
# - No, the retrieved chunks are all relevant to the question.



""" LlamaIndex Question 2 """
print("\t LlamaIndex Question 2\n")

question = [
    "What employee benefits does BrightLeaf offer?"
]
# set similarity_top_k=1 for the query engine
query_engine = index.as_query_engine(similarity_top_k=1)

print("\nset similarity_top_k=1 for the query engine\n")
print("=" * 30)
print(f"Q: {question[0]}")
print("=" * 30)
response = query_engine.query(question[0])
print("A:", response)
print("-" * 30)

for node_with_score in response.source_nodes:
    print(f"Node ID: {node_with_score.node.node_id}")
    print(f"Similarity Score: {node_with_score.score:.4f}")
    print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
    print("-" * 30)

# set similarity_top_k=3 for the query engine
query_engine = index.as_query_engine(similarity_top_k=3)

print("\nset similarity_top_k=3 for the query engine\n")
print("=" * 30)
print(f"Q: {question[0]}")
print("=" * 30)
response = query_engine.query(question[0])
print("A:", response)
print("-" * 30)

for node_with_score in response.source_nodes:
    print(f"Node ID: {node_with_score.node.node_id}")
    print(f"Similarity Score: {node_with_score.score:.4f}")
    print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
    print("-" * 30)

# Add a comment explaining how the response changed (if at all) and whether more retrieved context is always better.
# - The response with similarity_top_k=1 is more focused and concise. It takes exatly relevant info from the rigth document for this question.
# - The response with similarity_top_k=3 also good but it refrase the most relevant information, potentially including less relevant context.
# - More retrieved context is not always better, as it can introduce noise and less relevant information.



""" LlamaIndex Question 3 """
print("\t LlamaIndex Question 3\n")

question = [
    "What are the cutest animals in the world?"
]

query_engine = index.as_query_engine(similarity_top_k=3)

print("\nset similarity_top_k=3 for the query engine\n")
print("=" * 30)
print(f"Q: {question[0]}")
print("=" * 30)
response = query_engine.query(question[0])
print("A:", response)
print("-" * 30)

for node_with_score in response.source_nodes:
    print(f"Node ID: {node_with_score.node.node_id}")
    print(f"Similarity Score: {node_with_score.score:.4f}")
    print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
    print("-" * 30)


# what you expected, what actually happened, and what you would change about the system to handle this kind of query better.
# - I expect that the system would be strugle to find relevant info in the documents. 
# - What actually happened is that the system explicitly stated that it could not find relevant information in the documents 
# - and shorlty provided topics on what it could answer instead.
# - Actually, I woudln't change the system much, as it correctly identified that the information was not available in the documents 
# - and moreover, it provided alternative topics that it could answer, which is helpful for the user.

""" LlamaIndex Question 4 """
print("\t LlamaIndex Question 4\n")


# Create Judge LLM
llm = OpenAI(model="gpt-4o-mini", temperature=0.2)

# Define evaluator
faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
relevancy_evaluator = RelevancyEvaluator(llm=llm)

# Get response to query relevant to documents
q = "What employee benefits does BrightLeaf offer?"
response = query_engine.query(q)

print("=" * 30)
print("Q:", q)
print("A:", response)
print("=" * 30)

# Evaluate faithfulness and relevancy
faithfulness_result = faithfulness_evaluator.evaluate_response(query=q, response=response)
print("Faithfulness Evaluation: " + str(faithfulness_result.score))

relevancy_result = relevancy_evaluator.evaluate_response(query=q, response=response)
print("Relevancy Result: " + str(relevancy_result.score))

# Get response to query NOT relevant to documents
q = "What weather is expected for 2026-2027 winter season in NC USA?"
response = query_engine.query(q)

print("=" * 30)
print("Q:", q)
print("A:", response)
print("=" * 30)

# Evaluate faithfulness and relevancy
faithfulness_result = faithfulness_evaluator.evaluate_response(query=q, response=response)
print("Faithfulness Evaluation: " + str(faithfulness_result.score))

relevancy_result = relevancy_evaluator.evaluate_response(query=q, response=response)
print("Relevancy Result: " + str(relevancy_result.score))


# What does a faithfulness score of 1.0 mean? What would a score of 0.0 indicate?
# - A faithfulness score of 1.0 means the response is fully faithful to the retrieved context, 
# - while a score of 0.0 indicates the response contains hallucinations or lying. 
# What does a relevancy score measure, and how is it different from faithfulness?
# - A relevancy score measures how well the response addresses the query, regardless of whether it is faithful to the retrieved context. 
# - It is different from faithfulness, which only assesses alignment with the retrieved context.
# Did the scores change between your two queries? If so, why do you think that happened?
# - Yes, relavancy scores are opposite (1.0 for relevant queries, 0.0 for irrelevant queries)
# - while the faithfulness scores remain 1.0 for both queries, indicating that the responses are faithful 
# - despite the irrelevance of the second query. 
# What is the "LLM-as-a-judge" approach, and why is it used for RAG evaluation instead of a simple accuracy metric?
# - the evaluation metrics are computed using an external LLM (different from the RAG framework). 
# - An external LLM (trained appropriately) will be able to assess these subjective metrics faster compared to multiple human experts.
# - It is better than a simple accuracy metric, which only measures whether the response is correct, without considering faithfulness or relevancy.
 