# -----------------------------------------------------------------------
# Mini-Project — World Happiness Agent
# -----------------------------------------------------------------------


from dotenv import load_dotenv
from openai import OpenAI, api_key
import os
import json
from datetime import datetime

from pathlib import Path


import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr

# smolagents imports
from smolagents import Tool, ToolCallingAgent, OpenAIServerModel, tool
from smolagents import CodeAgent



if load_dotenv():
    print('Successfully loaded environment variables from .env')
else:
    print('Warning: could not load environment variables from .env')

client = OpenAI()
print('OpenAI client created.')

# ===========================================
# Pre-task: Load the Data
print("\nPre-task: Load the Data\n")
# ===========================================

print("happiness_project/    folder with files added\n ")

# initializing the empty global dataframe
df = None

# ===========================================
# Task 1: Define Your Tools
print("\nTask 1: Define Your Tools\n")
# ===========================================

""" Tool 1: load_happiness_data """

@tool
def load_happiness_data() -> dict:
    """ Load the World Happiness dataset into a global DataFrame df
    
        Reads CSV files from the 'happiness_project' directory (years 2015-2024), 
        parses semicolons as delimiters and commas as decimals, standardizes key 
        column names across different years, adds the extracted 'year' column, and 
        updates the shared global df. Returns a dict with shape and columns.

        Returns:
            dict: A dictionary containing:
            - 'shape' (list): The dimensions [rows, columns] of the merged DataFrame.
            - 'columns' (list): The list of column names in the merged dataset.
    """

    DATA_DIR = Path("happiness_project")

    if not DATA_DIR.exists():
        return {"error": f"Directory '{DATA_DIR}' not found."}
    
    COLUMN_NAMES = {
        "Country": "country",
        "Regional indicator": "region",
        "Happiness score": "happiness_score",
        "Ladder score": "happiness_score",   # the 2024 file uses this name
        "GDP per capita": "gdp_per_capita",
    }

    frames = []
    for path in sorted(DATA_DIR.glob("world_happiness_*.csv")):
        year_df = pd.read_csv(path, sep=";", decimal=",")
        year_df = year_df.rename(columns=COLUMN_NAMES)
        year_df["year"] = int(path.stem.split("_")[-1])   # "world_happiness_2020" -> 2020
        frames.append(year_df)

    if not frames:
        return {"error": "No CSV files found in 'happiness_project'."}

    global df
    df = pd.concat(frames, ignore_index=True)
    return {"shape": df.shape, "columns": df.columns.tolist()}



""" Tool 2: summarize_column """
@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for a single column in the loaded dataset.

        Args:
            column (str): The name of the column to summarize.

        Returns:
            dict: A dictionary containing the descriptive statistics for the specified column.
    """
    if df is None:
        return {"error": "No data loaded."}
    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}
    return df[column].describe().to_dict()


""" Tool 3: compute_correlation """
@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation coefficient and p-value between two numeric columns.

        Args:
            col1 (str): The name of the first column.
            col2 (str): The name of the second column.

        Returns:
            dict: A dictionary containing the correlation coefficient and p-value
                (rounded to 4 decimal places).
    """
    if df is None:
        return {"error": "No data loaded."}
    if col1 not in df.columns or col2 not in df.columns:
        return {"error": "One or both columns not found."}

    # Drop rows where either column has NaN to avoid calculation errors
    valid_data = df[[col1, col2]].dropna()

    if valid_data.empty:
        return {"error": f"No valid overlapping non-null data found between '{col1}' and '{col2}'."}

    try:
        pearson_r, p_value = pearsonr(valid_data[col1], valid_data[col2])
    except Exception as exc:
        return {"error": f"Failed to compute Pearson correlation: {str(exc)}"}

    return {"col1": col1, "col2": col2, "pearson_r": round(pearson_r, 4), "p_value": round(p_value, 4)}


""" Tool 4: get_top_n_countries """
@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """ Return the top n countries ranked by a given column for a specific year (if not specified, then top 5).
       
        Filters the loaded global dataset by the specified year, sorts the rows 
        in descending order based on the chosen column, and returns the highest 
        ranked countries along with their metric values.
        
        Args:
            column (str): The name of the column to rank by.
            year (int): The year for which to retrieve data.
            n (int): The number of top countries to return (default is 5).

        Returns:
            dict: A dictionary containing the top n countries and their values for the specified column.
    """
    if df is None:
        return {"error": "No data loaded."}
    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}
    if year not in df["year"].unique():
        return {"error": f"Year {year} not found."}
 
    # Filter data for the specified year
    year_df = df[df["year"] == year]
    top_df = year_df.dropna(subset=[column]).sort_values(by=column, ascending=False).head(n)

    top_records = top_df[["country", column]].to_dict(orient="records")

    return {
        "year": year,
        "column": column,
        "top_countries": top_records
    }


# ===========================================
# Task 2: Build the Agent
print("\nTask 2: Build the Agent\n")
# ===========================================

model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")

SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Note: the tools maintain the dataset internally. When you call `load_happiness_data()`,
it returns a dictionary with 'shape' and 'columns'. Read values directly from the tool's
returned dictionary rather than accessing `df` directly in your code.
Always use the provided tools where possible (load_happiness_data, summarize_column, 
compute_correlation, get_top_n_countries). Write Python code directly only when tools 
are insufficient (such as creating plots or custom aggregations).
Be concise and student-friendly in your responses.

IMPORTANT EXECUTION RULES:
1. Every response step MUST be formatted with thoughts followed by a code block:
   Thoughts: <your reasoning>
   <code>
   <your python code>
   </code>
2. When you have completed the analysis or have the final answer for the user, 
   you MUST return it using the built-in function `final_answer(...)` inside the code block:
   <code>
   final_answer("Your final concise explanation or summary here.")
   </code>
   NEVER write plain text or markdown outside of the designated code tags.
"""

agent = CodeAgent(
    tools=[load_happiness_data, summarize_column, compute_correlation, get_top_n_countries],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats"],
    max_steps=8,
)



# =======================================================================
# Running the Project
# =======================================================================

if __name__ == "__main__":
    # Checking if the output directory exists, if not, create it
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)


    # ===========================================
    # Task 3: Run Guided Queries
    print("\nTask 3: Run Guided Queries\n")
    # ===========================================

    queries = [
        "Load the happiness data and tell me its shape and column names.",
        "Summarize the happiness_score column.",
        "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
        "Show me the top 5 happiest countries in 2020.",
        "Plot happiness_score over the years as a line chart, with one line per region. Save the plot to outputs/happiness_by_region.png.",
    ]

    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = agent.run(query, reset=False, additional_args={"df": df})
        print(response)

    # ===========================================
    # Task 4: Your Own Questions
    print("\nTask 4: Your Own Questions\n")
    # ===========================================

    # My query 1
    my_query_1 = "Calculate the average happiness_score for each region in 2016 and sort the results from highest to lowest."   
    response_1 = agent.run(my_query_1, reset=False, additional_args={"df": df})
    print(response_1)
    # Comment: Did this trigger tool use, code generation, or both?
    # - No, It did't as I expected.

    # My query 2
    my_query_2 = "Identify the country with the highest GDP per capita in 2020."   
    response_2 = agent.run(my_query_2, reset=False, additional_args={"df": df})
    print(response_2)
    # Comment: Did this trigger tool use, code generation, or both?
    # - I expected that it would trigger tool use get_top_n_countries(n=1),
    # - but it didn't. Probably, it is because here we asked for just one country rather than providing a ranking. 

# ===========================================
# Task 5: Reflection
print("\nTask 5: Reflection\n")
# ===========================================

# --- Reflection ---
#
# 1. In Query 3, how did the agent report the correlation? The `compute_correlation` tool
#    returns both a Pearson r and a p-value — did the agent compute and communicate the
#    result sensibly (for example, describing the strength and direction of the
#    relationship, and noting how confident we can be in it)?
     # - yes, it used the tool correctly and explain the results by meaning 
     # - (indicates a moderate positive correlation via pearson_r, and signifies that 
     # - the correlation is statistically significant via p-value).
#
# 2. Did any of the agent's responses surprise you — either by being more capable than
#    you expected, or less? Describe one specific example.
#    - In natural-language queries asking for an extreme (e.g., "Identify the country 
#      with the highest GDP per capita in 2020"), the agent chose to bypass the specialized tool 
#      `get_top_n_countries` entirely and generated raw pandas code instead. It treated the task as 
#      an argmax calculation rather than realizing it could call `get_top_n_countries(..., n=1)`. 
#      While the result was correct, it revealed that CodeAgents inherently default to ad-hoc code 
#      generation unless explicitly nudged to leverage existing tools.

# 3. What one additional tool would make this agent meaningfully more useful?
#    Describe what it would do and what kind of question it would help the agent answer.
#    (You do not need to implement it.)
#      - Tool Name: `compare_countries_over_time`
#    - It accepts a list of country names (e.g., `countries: list[str]`) and a target metric 
#    - (e.g., `metric: str = 'happiness_score'`), filter the multi-year dataset across 2015-2024, 
#    - and return a structured year-over-year trajectory (including initial score, latest score, 
#    - net change, and peak year).
#    
#    What kind of question it would help answer:
#    It would allow the agent to answer comparative questions deterministically, 
#    such as: "How has `One country`'s happiness score evolved compared to `Another country` between 2015 and 2024, 
#    and which country experienced the largest improvement?" Currently, answering this requires 
#    the CodeAgent to write complex multi-step slicing and pivoting code, which is slower and 
#    more error-prone than a dedicated analytical tool.        