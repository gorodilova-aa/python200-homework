from dotenv import load_dotenv
from openai import OpenAI, api_key
import os
import json
from datetime import datetime

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# smolagents imports
from smolagents import ToolCallingAgent, OpenAIServerModel, tool
from smolagents import CodeAgent



if load_dotenv():
    print('Successfully loaded environment variables from .env')
else:
    print('Warning: could not load environment variables from .env')

client = OpenAI()
print('OpenAI client created.')

RESOURCES_DIR = Path("resources")
RESOURCES_DIR


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


class CsvManager:
    def __init__(self, resources_dir: Path):
        self.resources_dir = resources_dir
        self.df = None
        self.csv_name = None

    # --- Small internal helpers --------------------------------------

    def _normalize_csv_name(self, filename: str) -> str:
        if not filename.lower().endswith(".csv"):
            return filename + ".csv"
        return filename

    def _available_csv_files(self) -> list[str]:
        if not self.resources_dir.exists():
            return []
        return sorted(
            [
                p.name
                for p in self.resources_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".csv"
            ]
        )

    def _ensure_loaded(self):
        if self.df is None:
            files = self._available_csv_files()
            example = files[0] if files else "your_file.csv"
            return {
                "error": (
                    "No CSV is loaded yet. First load one from resources/. "
                    f"For example: load_csv '{example}'."
                )
            }
        return None

    # --- Tools (public methods) --------------------------------------

    def list_csv_files(self):
        """
        List available CSV files in resources/.
        """
        files = self._available_csv_files()
        if not files:
            return {
                "message": (
                    "No CSV files found in resources/. "
                    "Create a resources/ folder and put one or more .csv files inside it."
                ),
                "files": [],
            }
        return {"files": files}

    def load_csv(self, filename: str):
        """
        Load a CSV file from resources/ and make it the active dataset.

        filename can be "bike_commute" or "bike_commute.csv".
        """
        filename = self._normalize_csv_name(filename)
        path = self.resources_dir / filename

        if not path.exists():
            return {
                "error": f"Could not find '{filename}' in resources/.",
                "available_files": self._available_csv_files(),
            }

        self.df = pd.read_csv(path)
        self.csv_name = filename

        return {
            "message": f"Loaded {filename} with shape {self.df.shape}.",
            "columns": self.df.columns.tolist(),
        }

    def get_columns(self):
        """
        Return column names for the currently loaded CSV.
        """
        error = self._ensure_loaded()
        if error:
            return error
        return self.df.columns.tolist()

    def summarize_columns(self, columns: list[str] | None = None):
        """
        Return basic summary stats for one or more columns.

        If columns is None, summarize all columns.
        Uses pandas.describe(include="all") to stay simple and readable.
        """
        error = self._ensure_loaded()
        if error:
            return error

        if columns is None:
            data = self.df
        else:
            missing = [c for c in columns if c not in self.df.columns]
            if missing:
                return {"error": f"These columns are not in the data: {missing}"}
            data = self.df[columns]

        summary = data.describe(include="all").transpose().round(3)
        return summary.to_dict()

    def describe_column(self, column: str):
        """
        Simple summary for a single column using pandas.describe().
        """
        error = self._ensure_loaded()
        if error:
            return error

        if column not in self.df.columns:
            return {"error": f"'{column}' is not a column. Options: {self.df.columns.tolist()}"}

        s = self.df[column]
        summary = s.describe().to_dict()

        cleaned = {}
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                cleaned[key] = round(value, 3)
            else:
                cleaned[key] = value

        return cleaned

    def plot_data(self, y: str, x: str | None = None, plot_type: str = "line"):
        """
        Plot from the active CSV.
    
        - If x is None: plot y vs row index.
        - If x is provided: plot y vs x.
        """
        error = self._ensure_loaded()
        if error:
            return error
    
        if plot_type not in ["scatter", "line"]:
            return "Error: I can only do 'scatter' or 'line'."
    
        if y not in self.df.columns:
            return f"Error: column '{y}' is not in {self.df.columns.tolist()}"
    
        # If someone accidentally passes x == y, treat it like "plot y"
        if x == y:
            x = None
    
        # Scatter needs x
        if plot_type == "scatter" and x is None:
            return "Error: scatter plots need both x and y columns."
    
        title_csv = self.csv_name or "current CSV"
    
        if x is None:
            ax = self.df[y].plot(kind="line")
            ax.set_title(f"{title_csv} | Line plot: {y} vs row index")
            plt.show()
            return f"Plotted {y} vs row index as a line plot."
    
        if x not in self.df.columns:
            return f"Error: column '{x}' is not in {self.df.columns.tolist()}"
    
        ax = self.df.plot(x=x, y=y, kind=plot_type)
        ax.set_title(f"{title_csv} | {plot_type.title()} plot: {y} vs {x}")
        plt.show()
        
        return f"Plotted {y} vs {x} as a {plot_type}."

    def compute_correlation(self, col1: str, col2: str):
        """
        Compute the Pearson correlation between two columns in the loaded DataFrame.
        Returns a dictionary with  both columns, the correlation coefficient and p-value.
        """
        error = self._ensure_loaded()
        if error:
            return error

        if col1 not in self.df.columns:
            return {"error": f"'{col1}' is not a column. Options: {self.df.columns.tolist()}"}

        if col2 not in self.df.columns:
            return {"error": f"'{col2}' is not a column. Options: {self.df.columns.tolist()}"}

        from scipy.stats import pearsonr
        pearson_r, p_value = pearsonr(self.df[col1], self.df[col2])
        return {"col1": col1, "col2": col2, "pearson_r": round(pearson_r, 4), "p_value": round(p_value, 4)}

print("Class defined")


# Defining the tool schema
csv_backend = CsvManager(RESOURCES_DIR)

node_tools = {
    "list_csv_files": csv_backend.list_csv_files,
    "load_csv": csv_backend.load_csv,
    "get_columns": csv_backend.get_columns,
    "summarize_columns": csv_backend.summarize_columns,
    "describe_column": csv_backend.describe_column,
    "plot_data": csv_backend.plot_data,
    "compute_correlation": csv_backend.compute_correlation
}

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "list_csv_files",
            "description": "List available CSV files in the resources/ folder.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_csv",
            "description": "Load a CSV file from the resources/ folder and make it the active dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "CSV filename in resources/, e.g. 'bike_commute.csv'.",
                    }
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_columns",
            "description": "Get the column names of the currently loaded CSV.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_columns",
            "description": "Show basic summary statistics for columns (uses pandas.describe).",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of column names. If omitted, summarize all columns.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_column",
            "description": "Show basic summary statistics for a single column (uses pandas.describe).",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Column name to describe.",
                    }
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_data",
            "description": "Plot data from the active CSV. If only y is provided, plot y vs row index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "y": {"type": "string", "description": "Column name for y-axis."},
                    "x": {"type": "string", "description": "Optional column name for x-axis."},
                    "plot_type": {
                        "type": "string",
                        "enum": ["scatter", "line"],
                        "description": "Type of plot to create.",
                    },
                },
                "required": ["y"],
            },
        },
    },
    {
    "type": "function",
    "function": {
        "name": "compute_correlation",
        "description": "Compute the Pearson correlation between two columns in the loaded DataFrame. Returns the correlation coefficient and p-value.",
        "parameters": {
            "type": "object",
            "properties": {
                "col1": {
                    "type": "string",
                    "description": "The name of the first column to correlate."
                },
                "col2": {
                    "type": "string",
                    "description": "The name of the second column to correlate."
                }
            },
            "required": ["col1", "col2"]
        }
    }
}
]

# Defining a response cycle
def run_agent_cycle(messages, user_text, max_tool_rounds=5):
    """
    Run through one react-agent loop using a simple tool-using agent.
    `messages` parameter will usually just contain a system prompt, 
    and then user text will be appended.  

    The loop has three main steps:

    REASON:
      - Call the model with the conversation so far.
      - The model either replies normally, or asks to call a tool from tool set.

    ACT:
      - If tools are requested, run the Python functions

    OBSERVE:
      - Append each requested tool result back into the LLMs conversation history.
      - On the next iteration, the model reads those tool call results and determines
        whether it has reached the goal.

    Stop condition:
      - If the model returns an assistant message with no tool calls, this is the 
        final answer for this react cycle, this implies that reasoning alone without 
        tool calls was enough.  
      - max_tool_rounds is a safety cap to prevent infinite loops.
    """
    messages.append({"role": "user", "content": user_text})

    def observe_tool_result(tool_call_id, result):
        """
        Return a tool's return value as a message that can be appended to the
        LLMs conversation history. The model will read this tool output on the next
        REASON step.
        """
        content = json.dumps(result, default=str) if not isinstance(result, str) else result
        tool_message = {"role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": content,}
        return tool_message

    for loop_idx in range(max_tool_rounds):
        # REASON: call the model
        # Here it will make use of any previous tool outputs it appended ("observed")
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools_schema,
        )

        msg = response.choices[0].message

        # Append the assistant message to the conversation history.
        # Use a plain dict so `messages` stays simple and inspectable.
        assistant_entry = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
        messages.append(assistant_entry)

        # No tool calls means the model is answering directly.
        if not msg.tool_calls:
            return msg.content 

        # ACT + OBSERVE: run each tool call, then append its result.
        # Note there may be multiple tool calls
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments or "{}")

            print(f"ACT: {name}({tool_args})")

            fn = node_tools.get(name)
            if fn is None:
                result = {"error": f"Tool '{name}' not found."}
            else:
                try:
                    result = fn(**tool_args) if tool_args else fn()
                except Exception as e:
                    print(f"Tool error in {name}: {type(e).__name__}: {e}")
                    result = {"error": f"Tool '{name}' failed: {type(e).__name__}: {e}"}
                    
            # OBSERVE: append the tool result back into the conversation history.
            messages.append(observe_tool_result(tool_call.id, result))
            
            # After we appending information about all tool outputs, we loop back and REASON again.

    return "I hit the tool-round limit. Try a simpler request."




""" Q5 """
print("\t Q5\n")


SYSTEM_PROMPT = (
    "You are a small data assistant for CSV files stored in resources/. "
    "Use the available tools to do any data work (do not guess). "
    "If no CSV is loaded yet, load one first (or list available CSV files). "
    "Keep answers short and student-friendly."
)


def run_agent():
    """
    Simple command-line chat loop so it feels like a chatbot.

    We keep a single 'messages' list for the whole session so the model
    sees the conversation history each turn.
    """
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    print("CSV data agent at your service. Here to help look at your CSV data!")
    print("Type a question. Type 'exit' to quit.\n")
    print("To start, try 'list csv files' or 'load bike_commute.csv'\n")

    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in ["exit", "quit", "q"]:
            print("Bye.")
            break
        
        print(f"User query: {user_text}")
        assistant_text = run_agent_cycle(messages, user_text)
        print(f"\nAssistant: {assistant_text}\n")

run_agent()


# simulate the tool-round limit
"""
CSV data agent at your service. Here to help look at your CSV data!
Type a question. Type 'exit' to quit.

To start, try 'list csv files' or 'load bike_commute.csv'

You: list the csv files available to you, then load available file, then summarize the columns in the loaded csv file, then plot the relationship between the average speed and average heart rate, then What kind of correlation do you see between the columns you plotted?, then Plot the average speed against the average traffic density and ascertain the correlation between the two based on this
User query: list the csv files available to you, then load available file, then summarize the columns in the loaded csv file, then plot the relationship between the average speed and average heart rate, then What kind of correlation do you see between the columns you plotted?, then Plot the average speed against the average traffic density and ascertain the correlation between the two based on this
ACT: list_csv_files({})
ACT: load_csv({'filename': 'bike_commute.csv'})
ACT: summarize_columns({})
ACT: plot_data({'x': 'avg_heart_rate', 'y': 'avg_speed_kmh', 'plot_type': 'scatter'})
ACT: compute_correlation({'col1': 'avg_speed_kmh', 'col2': 'avg_heart_rate'})
ACT: plot_data({'x': 'avg_traffic_density', 'y': 'avg_speed_kmh', 'plot_type': 'scatter'})
ACT: compute_correlation({'col1': 'avg_speed_kmh', 'col2': 'avg_traffic_density'})

Assistant: I hit the tool-round limit. Try a simpler request.

"""


# set up the agent with the system prompt from the lesson, then run:
messages = [{"role": "system", "content": SYSTEM_PROMPT}]
result = run_agent_cycle(messages, "Load bike_commute.csv and compute the correlation between avg_traffic_density and avg_speed_kmh.")
print(result)

# Print the agent's final response
"""
ACT: list_csv_files({})
ACT: load_csv({'filename': 'bike_commute.csv'})
ACT: compute_correlation({'col1': 'avg_traffic_density', 'col2': 'avg_speed_kmh'})
The correlation between average traffic density and average speed (km/h) is approximately -0.53, indicating a moderate negative relationship. 
As traffic density increases, average speed tends to decrease. This result is statistically significant (p-value = 0.0).
"""


""" Q6 """
print("\t Q6\n")

# Role mapping in the ReAct (Reason -> Act -> Observe) loop:
# - 'system': Sets the overall behavior, boundaries, instructions, and tool descriptions for the agent.
# - 'user': Provides the objective/prompt (the task or goal the agent needs to accomplish).
# - 'assistant': Represents the model's reasoning ("Reason") and decision to call a tool ("Act"), 
#                or the final answer to the user once the goal is reached.
# - 'tool': Represents the environment feedback/execution output ("Observe"), returning the tool's 
#           result back to the model so it can evaluate what to do next.

print(json.dumps(messages, indent=2, default=str))


# ------------------------------------------------------------------------------------
# --- Lesson 04: smolagents --- 
print("\n ---  Lesson 04: smolagents ---\n")
# ------------------------------------------------------------------------------------

""" Q7 """
print("\t Q7\n")

csv_manager = CsvManager(resources_dir=RESOURCES_DIR)
# wraping the CsvManager methods into smolagents tools. 

@tool
def list_csv_files() -> dict:
    """List available CSV files in resources/.

    Returns:
        A dict with a "files" list, or a message if none are found.
    """
    return csv_manager.list_csv_files()


@tool
def load_csv(filename: str) -> dict:
    """Load a CSV file from resources/ and make it the active dataset.

    Args:
        filename: CSV filename in resources/. You can pass "bike_commute" or "bike_commute.csv".

    Returns:
        A dict with a status message and column names, or an error dict.
    """
    return csv_manager.load_csv(filename)


@tool
def get_columns() -> list[str] | dict:
    """Return column names for the currently loaded CSV.

    Returns:
        A list of column names, or an error dict if no CSV is loaded.
    """
    return csv_manager.get_columns()


@tool
def summarize_columns(columns: list[str] | None = None) -> dict:
    """Return summary stats for selected columns (or all columns). 
    This includes count, mean, std, min, max, and percentiles for numeric columns,
    or count, unique, top, freq for categorical columns.

    Args:
        columns: Column names to summarize. If None, summarizes all columns.

    Returns:
        A dict of summary statistics (from pandas.describe), or an error dict.
    """
    return csv_manager.summarize_columns(columns)


@tool
def describe_column(column: str) -> dict:
    """Describe a single column (basic stats) for the requested column.
    This includes count, mean, std, min, max, and percentiles for numeric column,
    or count, unique, top, freq for categorical column.

    Args:
        column: The name of the column to describe.

    Returns:
        A dict of basic stats for the column, or an error dict.
    """
    return csv_manager.describe_column(column)


@tool
def plot_data(y: str, x: str | None = None, plot_type: str = "line") -> str | dict:
    """Plot from the active CSV.

    Args:
        y: Column name to plot on the y-axis. 
        x: Column name to plot on the x-axis. If None, use row index.
        plot_type: "line" or "scatter". Scatter requires x and y.

    Returns:
        Generates and shows the plot. 
        Retirms a short success message string, or an error dict/string.
    """
    return csv_manager.plot_data(y=y, x=x, plot_type=plot_type)

@tool 
def compute_correlation(col1: str, col2: str) -> str | dict:
    """Compute the Pearson correlation coefficient and p-value between two numeric columns.

    Args:
        col1: The name of the first column.
        col2: The name of the second column.

    Returns:
       Returns the correlation coefficient and p-value (or an error message).
    """
    return csv_manager.compute_correlation(col1, col2)

# define the tools list for the agents.
TOOLS = [
    list_csv_files,
    load_csv,
    get_columns,
    summarize_columns,
    describe_column,
    plot_data,
    compute_correlation
]


print(compute_correlation.description)

# Comparison: smolagents auto-generated schema vs. manual JSON schema (Q4)
#
# 1. Structural Comparison:
#    - In Q4, we manually constructed a verbose, nested JSON Schema specifying the tool's 
#      "name", top-level "description", "parameters" object, each property's JSON type 
#      ("string"), description, and the "required" array.
#    - smolagents generates this exact specification automatically behind the scenes by 
#      introspecting the Python function signature, type annotations, and docstring, 
#      producing an equivalent schema without manual boilerplate.
#
# 2. What smolagents needs from the developer to produce a quality tool description:
#    - Python Type Annotations (`col1: str, col2: str -> dict`):
#      smolagents maps these directly to JSON Schema parameter types (e.g., str -> string, 
#      int/float -> number, dict -> object). Without them, the agent cannot validate inputs.
#    - Clear, Descriptive Docstring (Google or Sphinx style):
#      - Main description: Tells the model *what* the tool does and *when* to invoke it.
#      - `Args:` block: Details *what each parameter represents*, expected constraints, or formats.
#      - `Returns:` section: Explains the structure and output the model should anticipate.


""" Q8 """
print("\t Q8\n")


model_to_use = "gpt-4o-mini"  # default model ID
model = OpenAIServerModel(
    api_key=api_key,
    model_id=model_to_use,
)


# tool agent
SYSTEM_PROMPT = (
    "You are a small data assistant to help analyze files stored in resources/. "
    "Use the available tools to do any work requested (do not guess). "
    "Keep answers short and student-friendly."
)

tool_agent = ToolCallingAgent(tools=TOOLS,
                         model=model,
                         instructions=SYSTEM_PROMPT,)




# code agent
CODE_INSTRUCTIONS = """
You are a helpful CSV analysis assistant.

You can do two kinds of actions:
1) Call the provided tools.
2) Write and execute Python code when tools are not enough.

Rules:
- Prefer tools for simple tasks.
- IMPORTANT: If the user requests plot styling (color, marker, title text, labels, grid, etc.)
  that the plot_data tool cannot control, DO NOT call plot_data.
  Instead, write matplotlib code directly so the plot matches the request.
  If code execution fails, do not fall back to plot_data when the user requested styling (like color). 
  Explain what failed and what you would need to proceed.
- Be honest: only claim you did something if the code or tool actually did it.
- Assume the active dataset lives in csv_manager.df after a CSV is loaded.
"""

code_agent = CodeAgent(
    tools=TOOLS,
    model=model,
    instructions=CODE_INSTRUCTIONS,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "numpy"],
    max_steps=8,
)

# prompt for tool and code agents
prompt = "Load bike_commute.csv. Plot avg_heart_rate vs duration_min as a scatter plot with green dots."

response_tool = tool_agent.run(prompt)
response_code = code_agent.run(prompt, additional_args={"csv_manager": csv_manager})


# 1. What did each agent actually produce? Did the ToolCallingAgent change the dot color? Did the CodeAgent?
#    - ToolCallingAgent: 
#      Called the pre-defined `plot_data` tool from CsvManager. Because that tool's hardcoded 
#      implementation (or schema) does not accept custom styling arguments like dot color, 
#      the agent could not modify the color—it produced the default scatter plot (blue dots). 
#      It was constrained strictly by the fixed interface of the provided tool.
#    - CodeAgent: 
#      Wrote and executed custom Python code (using `matplotlib.pyplot`). Instead of being 
#      boxed in by existing tool signatures, it extracted the dataframe columns, called 
#      `plt.scatter(..., color='green')`, and successfully produced a plot 
#      with green dots exactly as requested.
#
# 2. What does this reveal about when each type of agent is more useful?
#    - ToolCallingAgent is best for:
#      - Strict, deterministic, and safe workflows where actions must stay within rigid boundaries 
#        (e.g., executing predefined database queries, triggering transactional APIs, or sending emails).
#      - Environments where arbitrary code execution is a security vulnerability or unsupported.
#      - Simple, single-step tasks that fit neatly into static tool signatures.
#
#    - CodeAgent is best for:
#      - Exploratory data analysis (EDA), visualization, and ad-hoc data manipulation where flexibility is key.
#      - Complex logic involving loops, conditionals, transformations, or combining multiple data sources 
#        that would otherwise require dozens of specialized, rigid tool definitions.
#      - Tasks where you need to adapt presentation details (e.g., plot styles, formatting, multi-line plots) 
#        on the fly without redesigning the backend tools.


""" Q9 """
print("\t Q9\n")

# 1. Describe a task where a ToolCallingAgent would be a better choice than a CodeAgent. 
#    What property of the task makes it a good fit for a tool-based approach?
#    - Task: A customer support agent processing refunds or order cancellations via an API.
#    - Key Property: High determinism and strict boundaries. The task requires fixed, discrete 
#      actions with validated arguments, where custom algorithmic logic or data transformation 
#      is unnecessary and risky.
#
# 2. What is one meaningful risk of using a CodeAgent that does not apply to a ToolCallingAgent? 
#    (Think about what's actually happening when the agent generates and runs code.)
#    - Risk: Arbitrary code execution / Security vulnerabilities.
#    - Detail: A ToolCallingAgent is physically restricted to predefined functions and schemas. 
#      A CodeAgent dynamically generates and executes raw Python code via an interpreter, 
#      which introduces severe risks if prompted maliciously or hallucinating—such as deleting 
#      files, accessing environment variables/secrets, or running infinite loops.