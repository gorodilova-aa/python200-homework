from dotenv import load_dotenv
from openai import OpenAI
import json

# ------------------------------------------------------------------------------------
# --- The Chat Completions API --- 
print("\n ---  The Chat Completions API ---\n")
# ------------------------------------------------------------------------------------

load_dotenv()
client = OpenAI()

""" API Question 1 """
print("\t API Question 1\n")

# response creation 
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
)

# extract the response text, model name, and total tokens used
response_text = response.choices[0].message.content
model_name = response.model
total_tokens = response.usage.total_tokens

# print the response text, model name, and total tokens used
print(f"Response:\n{response_text}\n")
print(f"Model: {model_name}")
print(f"Total tokens used: {total_tokens}\n")



""" API Question 2 """
print("\t API Question 2\n")

# setting a prompt and varying the temperature parameter
prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

# iterate over the different temperature values and generate responses for each
for temp in temperatures:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temp
    )
    response_text = response.choices[0].message.content
    model_name = response.model
    total_tokens = response.usage.total_tokens

    print(f"Response (Temperature: {temp}):\n{response_text}\n")
    print(f"Model: {model_name}")
    print(f"Total tokens used: {total_tokens}\n")

# What do you notice about how the outputs differ? 
# - The outputs become more creative and varied as the temperature increases.
# Which temperature would you use if you needed a consistent, reproducible output?
# - A temperature of 0-0.2 would be more consistent and reproducible.


""" API Question 3 """
print("\t API Question 3\n")

# generate multiple responses for the same prompt using the n parameter
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
    n=3,
    temperature=1.0
)

# print the multiple responses generated
for i, choice in enumerate(response.choices, start=1):
    print(f"Response {i}:\n{choice.message.content}\n")



""" API Question 4 """
print("\t API Question 4\n")

# setting a promt and using the max_tokens parameter to limit the length of the response
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain how neural networks work."}],
    max_tokens=15,
)

response_text = response.choices[0].message.content
model_name = response.model
completion_tokens = response.usage.completion_tokens

print(f"Response:\n{response_text}\n")
print(f"Model: {model_name}")
print(f"Completion tokens used: {completion_tokens}\n")

# What happened? 
# - The response was truncated to 15 tokens, as specified by the max_tokens parameter.
# - the sentence was cut off and incomplete. 
# Why might you want to use max_tokens in a real application?
# - 1. To control costs by limiting the number of tokens generated.
# - 2. To ensure responses fit within a certain length for display or processing.
# - 3. To prevent excessively long outputs that could overwhelm users or systems.




# ------------------------------------------------------------------------------------
# --- System Messages and Personas --- 
print("\n ---  System Messages and Personas ---\n")
# ------------------------------------------------------------------------------------

""" System Question 1 """
print("\t System Question 1\n")

# 1) Define the system message and user message for the conversation.
messages = [
    {"role": "system", "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
)

response_text = response.choices[0].message.content
print("\n-------------\n")
print(f"System content:\n{messages[0]['content']}\n")
print(f"User content:\n{messages[1]['content']}\n")
print(f"Response:\n{response_text}\n")


# 2) Changing the system message for the same user message for the conversation.
messages = [
    {"role": "system", "content": "You doesn't know about Python. You are a biologist who is trying to learn Python. You are confused and ask many questions."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
)

response_text = response.choices[0].message.content
print("\n-------------\n")
print(f"System content:\n{messages[0]['content']}\n")
print(f"User content:\n{messages[1]['content']}\n")
print(f"Response:\n{response_text}\n")


# Add a comment noting what changed.
# - The response for the first system message was more explanatory and encouraging, 
# - while the response for the second system message was shorter (although still correct) and ends with questioning.


""" System Question 2 """
print("\t System Question 2\n")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
)

response_text = response.choices[0].message.content
print(f"Response:\n{response_text}\n")

# Why does the model know Jordan's name, even though it's stateless?
# - The model knows Jordan's name because it is provided in the conversation history within the messages list. 




# ------------------------------------------------------------------------------------
# --- Prompt Engineering --- 
print("\n ---  Prompt Engineering ---\n")
# ------------------------------------------------------------------------------------

""" Prompt Question 1 — Zero-Shot """
print("\t Prompt Question 1 — Zero-Shot \n")

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

for i, review in enumerate(reviews, start=1):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": f"classify the sentiment of each review below as positive, negative, or mixed.\nReview: {review}"}
        ]
    )
    response_text = response.choices[0].message.content
    print(f"Review {i}: {response_text}\n")
  


""" Prompt Question 2 — One-Shot """
print("\t Prompt Question 2 — One-Shot \n")


reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

# prompt with one example review and its sentiment classification
for i, review in enumerate(reviews, start=1):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": f"""classify the sentiment of each review below as positive, negative, or mixed. Review: {review}.
                
                Example:
                Review: "Fast shipping but the item arrived damaged."
                Sentiment: mixed 
                """}
        ]
    )
    response_text = response.choices[0].message.content
    print(f"Review {i}: {response_text}\n")


# Did adding one example change the format or consistency of the output compared to Q1?
# - Yes, adding one example improved the consistency of the output format. 
# - It uses "Sentiment: ..." for all three reviews, whereas in Q1, the output varied in format.  

""" Prompt Question 3 — Few-Shot """
print("\t Prompt Question 3 — Few-Shot \n")


reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

# prompt with three example reviews and their sentiment classification
for i, review in enumerate(reviews, start=1):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": f"""classify the sentiment of each review below as positive, negative, or mixed. Review: {review}.
                
                Example 1:
                Review: "Fast shipping but the item arrived damaged."
                Sentiment: mixed
                
                Example 2:
                Review: "Great customer service and very helpful staff."
                Sentiment: positive
                
                Example 3:
                Review: "The product was in bad condition seems to be used before."
                Sentiment: negative
                """}
        ]
    )
    response_text = response.choices[0].message.content
    print(f"Review {i}: {response_text}\n")


# comparing all three approaches (zero-shot, one-shot, few-shot): When would you choose each one?
# - Zero-shot: When you need a quick straightforward answer.
# - One-shot: When you want to provide a single example to guide the model's consistency in output format or style.
# - Few-shot: when you want to provide multiple examples to improve the model's understanding of the task and ensure more accurate and consistent outputs.


""" Prompt Question 4 — Chain of Thought"""
print("\t Prompt Question 4 — Chain of Thought\n")


prompt = """
Solve the following problem step by step 
paying attention to the detailed reasoning before giving a final answer. 
Label the final answer clearly.

Problem:
A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later
takes a new job that pays $7,500 more per year than her post-raise salary.
What is her final annual salary?
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)
response_text = response.choices[0].message.content
print(f"Chain of Thought Response: {response_text}\n")

# Why does asking the model to reason step by step tend to improve accuracy on problems like this?
# - You are guiding the model to break down the problem into smaller, manageable steps,
# - which helps ensure that it considers all relevant information and calculations before arriving at a final answer.       
# - Also, you can more easily identify the correctness of each step and final answer. 




""" Prompt Question 5 — Structured Output """
print("\t Prompt Question 5 — Structured Output\n")


review = "I've been using this tool for three months. It handles large datasets well, \
but the UI is clunky and the export options are limited."


prompt = f"""Analyze the sentiment of the review below.
Respond ONLY as a valid JSON object (no markdown, no code fences, no extra text) with exactly the following keys:
- "sentiment": "positive", "negative", or "mixed"
- "confidence": a float between 0 and 1 representing your certainty
- "reason": one sentence explaining your result

Review: "{review}"
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    response_format={"type": "json_object"},
)

raw_response = response.choices[0].message.content

print(f"Raw response:\n{raw_response}\n")

try:
    data = json.loads(raw_response)
    print("Parsed JSON fields:")
    print(f"Sentiment: {data.get('sentiment')}")
    print(f"Confidence: {data.get('confidence')}")
    print(f"Reason: {data.get('reason')}")
except json.JSONDecodeError as e:
    print(f"Failed to parse JSON: {e}")
    print(f"Raw response for debugging:\n{raw_response}")


""" Prompt Question 6 — Delimiters"""
print("\t Prompt Question 6 — Delimiters\n")


user_text = "First boil a pot of water. Once boiling, add a handful of salt and the \
pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
)

response_text = response.choices[0].message.content

print(f"Response 1:\n{response_text}\n")


# checking the model's response to a different input that does not contain instructions
user_text_2 = "Novosibirsk is the third most populous city in Russia, located in southwestern Siberia. \
    It is an important industrial and cultural center in the region."

prompt_2 = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text_2}```
"""

response_2 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt_2}],
)

response_text_2 = response_2.choices[0].message.content

print(f"Response 2:\n{response_text_2}\n")


# What problem do delimiters help prevent?
# - Delimiters help prevent the model from misinterpreting the input text or instructions.
# - They clearly define the boundaries of the input.

# ------------------------------------------------------------------------------------
# --- Local Models with Ollama --- 
print("\n ---  Local Models with Ollama ---\n")
# ------------------------------------------------------------------------------------

""" Ollama Question 1 """
print("\t Ollama Question 1\n")

"""
We run the following command in the terminal: 
ollama run qwen3:0.6b "Explain what a large language model is in two sentences."

Response from Ollama:
Thinking...
Okay, the user wants me to explain what a large language model is in two sentences. Let me start by recalling what I know. A large language model is a type of model that can 
understand and generate human-like text. It's used for tasks like language translation, text generation, and answering questions. I should mention the key features like 
understanding multiple languages and generating contextually relevant text. Need to make sure both sentences are concise and cover the main points without being too technical.
...done thinking.

A large language model is a complex system that can understand and generate human-like text, enabling it to perform tasks like translation, text generation, or answering 
questions. It relies on vast amounts of training data and uses deep neural networks to learn patterns and context from this information.

"""

# response creation for OpenAI model
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain what a large language model is in two sentences."}]
)

response_text = response.choices[0].message.content
print(f"OpenAI API Response:\n{response_text}\n")

# What differences did you notice between the two responses? 
# - The Ollama response included a "thinking" process and was more verbose, 
# - explaining the steps it took to arrive at the answer. 
# - The OpenAI API response was more concise and directly answered the question without additional context in two sentences as requested.
# What is one advantage and one disadvantage of running a model locally?
# - Advantage: Running a model locally can provide faster response times and greater control over the model's behavior and data privacy.
# - Disadvantage: It may require significant computational resources 
# - and may not have access to the latest updates or improvements available in cloud-hosted models.
