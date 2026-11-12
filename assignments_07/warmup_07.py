from dotenv import load_dotenv
from openai import OpenAI
import os
import json
from datetime import datetime

if load_dotenv():
    print('Successfully loaded environment variables from .env')
else:
    print('Warning: could not load environment variables from .env')

client = OpenAI()
print('OpenAI client created.')


# ------------------------------------------------------------------------------------
# --- Lesson 02: Tool Definitions and the ReAct Loop --- 
print("\n ---  Lesson 02: Tool Definitions and the ReAct Loop ---\n")
# ------------------------------------------------------------------------------------

""" Q1 """
print("\t Q1\n")

# tool function for converting Celsius to Fahrenheit
def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"

# JSON schema for the tools list
tools = [
    {
        'type': 'function',
        'function': {
            'name': 'celsius_to_fahrenheit',
            'description': 'Converts a Celsius temperature (float) to Fahrenheit and returns it as a formatted string.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'celsius': {
                        'type': 'number',
                        'description': 'The temperature in Celsius.'
                    }
                },
                'required': ['celsius'],
            },
        },
    }
]
print('Tools list defined with one tool: celsius_to_fahrenheit\n')

# Call the function
print(celsius_to_fahrenheit(0))
print(celsius_to_fahrenheit(100))
print(celsius_to_fahrenheit(-40) + "\n")


""" Q2 """
print("\t Q2\n")


# Function copied from the lesson 
def run_agent(user_prompt: str) -> str:
    '''Run a minimal ReAct-style agent for a single user prompt.'''

    SYSTEM_PROMPT = '''You are a simple assistant that can tell the current time.
                     Use the tool get_current_time whenever a user asks about the time.'''
    
    # Step 1: start the conversation with system and user messages
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]

    # Step 2: first API call - the model decides whether to call a tool
    first_response = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=messages,
        tools=tools,
        tool_choice='auto',  # model chooses whether to use a tool
    )

    print("First response received from model...")
    print(first_response)
    first_message = first_response.choices[0].message

    # Record what the model said so far
    messages.append(
        {
            'role': 'assistant',
            'content': first_message.content,
            'tool_calls': first_message.tool_calls,
        }
    )

    # Step 3: check if the model requested any tools
    if first_message.tool_calls:
        print("Agentic mode engaged...")
        for tool_call in first_message.tool_calls:
            function_name = tool_call.function.name
            # In this example we only have one tool: get_current_time
            if function_name == 'get_current_time':
                tool_result = get_current_time()
            else:
                tool_result = f'Error: unknown tool {function_name}.'

            # Print for debugging so we can see what happened
            print('Tool called:', function_name)
            print('Tool result:', tool_result)

            # Step 3b: append the tool output so the model can see it
            messages.append(
                {
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'name': function_name,
                    'content': tool_result,
                }
            )

        # Step 4: second API call - model sees the tool result and gives final answer
        second_response = client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=messages,
        )
        print("Second response received from model...")
        print(second_response)

        final_message = second_response.choices[0].message
        return final_message.content or ''
    else:
        print("No tools needed....")

    # If there were no tool calls, the first response was already the final answer
    return first_message.content or ''


# Will calling run_agent("Convert 100 degrees Celsius to Fahrenheit") trigger a tool call? Why or why not?
# - No, because we didn`t define in SYSTEM_PROMPT that there is a tool for temperature conversion.
# How many API calls will be made to answer this query?
# - One, because the model will not call any tools and will directly provide the answer.

run_agent("Convert 100 degrees Celsius to Fahrenheit")

# Surprisingly, I was not right. This query triggered a tool call. 
# But since we did not specified how to handle the tool call, the model did not execute it.
# But the second query still present and assistant wrote a respose without a tool. 
# This shows that the model can still provide an answer even if a tool call is triggered but not executed.


""" Q3 """
print("\t Q3\n")

# Define the tool for getting the current time
def get_current_time() -> str:
    '''Return the current local time as a formatted string.'''
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# tool function for converting Celsius to Fahrenheit
def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"

# updated JSON schema for the tools
tools_extended = [
    {
        'type': 'function',
        'function': {
            'name': 'get_current_time',
            'description': 'Returns the current local time as a string.',
            'parameters': {
                'type': 'object',
                'properties': {},
                'required': [],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'celsius_to_fahrenheit',
            'description': 'Converts a Celsius temperature (float) to Fahrenheit and returns it as a formatted string.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'celsius': {
                        'type': 'number',
                        'description': 'The temperature in Celsius.'
                    }
                },
                'required': ['celsius'],
            },
        },
    }
]
print('Tools list NOW defined with two tools: get_current_time and celsius_to_fahrenheit\n')

# Extended version of the run_agent function 
def run_agent_extended(user_prompt: str) -> str:
    '''Run a minimal ReAct-style agent for a single user prompt.'''

    SYSTEM_PROMPT = '''You are a simple assistant that can:
                        -   tell the current time.
                            Use the tool get_current_time whenever a user asks about the time.
                        -   convert celsius to fahrenheit.
                            Use the tool celsius_to_fahrenheit whenever a user asks for this conversion.'''
    
    # Step 1: start the conversation with system and user messages
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]

    # Step 2: first API call - the model decides whether to call a tool
    first_response = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=messages,
        tools=tools_extended,
        tool_choice='auto',  # model chooses whether to use a tool
    )

    print("First response received from model...")
    print(first_response)
    first_message = first_response.choices[0].message

    # Record what the model said so far
    messages.append(
        {
            'role': 'assistant',
            'content': first_message.content,
            'tool_calls': first_message.tool_calls,
        }
    )

    # Step 3: check if the model requested any tools
    if first_message.tool_calls:
        print("Agentic mode engaged...")
        for tool_call in first_message.tool_calls:
            function_name = tool_call.function.name
            # In this example we have two tools: get_current_time and celsius_to_fahrenheit
            if function_name == 'get_current_time':
                tool_result = get_current_time()
            elif function_name == 'celsius_to_fahrenheit':
                args = json.loads(tool_call.function.arguments)
                celsius = args.get('celsius')
                tool_result = celsius_to_fahrenheit(celsius)
            else:
                tool_result = f'Error: unknown tool {function_name}.'

            # Print for debugging so we can see what happened
            print('Tool called:', function_name)
            print('Tool result:', tool_result)

            # Step 3b: append the tool output so the model can see it
            messages.append(
                {
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'name': function_name,
                    'content': tool_result,
                }
            )

        # Step 4: second API call - model sees the tool result and gives final answer
        second_response = client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=messages,
        )
        print("Second response received from model...")
        print(second_response)

        final_message = second_response.choices[0].message
        return final_message.content or ''
    else:
        print("No tools needed....")

    # If there were no tool calls, the first response was already the final answer
    return first_message.content or ''

# cheking the extended agent 
print("\n" +"*" * 30)
response_a = run_agent_extended("What is 37 degrees Celsius in Fahrenheit?")
print("Response A:", response_a)

# - tool was called and correctly executed
# - why? we specified in the SYSTEM_PROMPT that the agent should use the celsius_to_fahrenheit tool for such conversions.

print("\n" +"*" * 30)
response_b = run_agent_extended("What is the boiling point of water in plain English?")
print("Response B:", response_b)

# - tool was not called
# - why? the question did not require any tool usage, so the agent answered directly.


# ------------------------------------------------------------------------------------
# --- Lesson 03: Multi-Tool Agent --- 
print("\n ---  Lesson 03: Multi-Tool Agent ---\n")
# ------------------------------------------------------------------------------------

""" Q4 """
print("\t Q4\n")

""" Q5 """
print("\t Q5\n")

""" Q6 """
print("\t Q6\n")




# ------------------------------------------------------------------------------------
# --- Lesson 04: smolagents --- 
print("\n ---  Lesson 04: smolagents ---\n")
# ------------------------------------------------------------------------------------

""" Q7 """
print("\t Q7\n")

""" Q8 """
print("\t Q8\n")

""" Q9 """
print("\t Q9\n")