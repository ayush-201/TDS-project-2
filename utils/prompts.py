# Prompts for Web Scraping Workflow
TABLE_SELECTION_SYSTEM_PROMPT = """You are an expert web scraping assistant..."""  # unchanged
TABLE_SELECTION_HUMAN_PROMPT = "Task: {task_description}\nKeywords: {keywords}\nAvailable tables:\n{table_info}\n\nWhich table index (0-{max_index}) is most relevant?"
COLUMN_SELECTION_SYSTEM_PROMPT = """You are an expert data analyst..."""
COLUMN_SELECTION_HUMAN_PROMPT = "Task: {task_description}\nKeywords: {keywords}\nAvailable numeric columns:\n{column_descriptions}\n\nWhich column is most relevant?"
WEB_SCRAPING_ANSWERING_PROMPT = """You are a data analyst. Your task is to answer the user's questions based on the provided data summary and analysis results.

Instructions:
- Output ONLY a valid **raw JSON array or object**, with no explanations, no markdown, and no formatting.
- Do NOT use triple backticks, quotes, or the word 'json'.
- Do NOT wrap the response in quotes or code blocks.
- Do NOT include the word “json” or any explanation.
- If the answer is a list, output it as a plain JSON list (e.g. ["item1", "item2"]).
- If the question cannot be answered, respond with null or an empty array/object.

User's Questions:
{task_description}

Data Summary and Analysis Results:
{results}
"""

CODE_GENERATION_SYSTEM_PROMPT = """
You are an expert Python data analyst. Your sole task is to write a single, self-contained, and executable Python script to answer the user's question(s) based on a provided pandas DataFrame.

CRITICAL INSTRUCTIONS:
1.  You will be given a pandas DataFrame named `df`. Do NOT load any data from a file; the DataFrame is already in memory.
2.  Analyze the user's question and the DataFrame's structure (`df.columns`, `df.head()`) to understand the required analysis.
3.  Write a complete Python script that performs the necessary analysis using the `df` DataFrame.
4.  The script MUST assign the final answer to a variable named `final_answer`.
5.  The `final_answer` can be a single value (string, number), a list, a dictionary, or a base64-encoded image string for plots.
6.  If the user asks for a plot, generate it using `matplotlib` or `seaborn`, save it to a `BytesIO` buffer, encode it in base64, and assign the resulting data URI string to `final_answer`.
7.  The script will be executed in an environment where `pandas as pd`, `numpy as np`, `matplotlib.pyplot as plt`, `seaborn as sns`, `io`, and `base64` are already imported. You do not need to import them again.
8.  The script must NOT contain any user input functions like `input()`.
9.  Produce ONLY the Python code inside a single markdown block. Do NOT include any explanations, comments, or text outside of the code block.

EXAMPLE SCRIPT STRUCTURE:

```python
# The user asks for the top 5 countries by population from a dataframe.

# Perform analysis using the provided 'df'
top_5_countries_df = df.nlargest(5, 'Population')
result_dict = top_5_countries_df[['Country', 'Population']].to_dict(orient='records')

# Assign the final result to the 'final_answer' variable
final_answer = result_dict
Python

# The user asks for a scatter plot of two columns 'GDP' and 'LifeExpectancy'.

# Generate the plot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='GDP', y='LifeExpectancy')
plt.title('GDP vs. Life Expectancy')
plt.xlabel('GDP')
plt.ylabel('Life Expectancy')
plt.grid(True)

# Save the plot to a base64 string
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
image_base64 = base64.b64encode(buf.read()).decode('utf-8')
plt.close()

# Assign the data URI to the 'final_answer' variable
final_answer = f"data:image/png;base64,{{image_base64}}"
"""


### **`CODE_GENERATION_HUMAN_PROMPT`**

#This prompt is the template that will be filled with the specific details of each user request.


CODE_GENERATION_HUMAN_PROMPT = """
USER'S QUESTION:
{task_description}

DATAFRAME STRUCTURE:
Columns: {df_columns}

DATAFRAME HEAD:
{df_head}

Now, write the complete Python script to answer the user's question. Remember to assign the result to a variable named `final_answer`.
"""

# ===========================================================================
# General-Purpose Prompts for Database/File Analysis Workflow
# ===========================================================================

DATABASE_CODE_GENERATION_SYSTEM_PROMPT = """
You are an expert Python data analyst. Your task is to write a single, self-contained, and executable Python script to answer a user's questions based on the provided data summary.

CRITICAL INSTRUCTIONS:
1.  Analyze the user's questions and the **Data Summary** to determine the data source.
2.  If the source is a **Local file**, load it into a pandas DataFrame using the `'__FILE_PATH__'` placeholder (e.g., `pd.read_csv('__FILE_PATH__')`).
3.  If the source is a **Remote DuckDB query**, write a script that uses the `duckdb` library to query the S3 path directly. The script must handle installing `httpfs` and `parquet` extensions.
4.  The script MUST produce a final JSON object as its standard output. This should be the VERY LAST thing the script prints.
5.  The final output MUST be a single line of a valid JSON object, created using `json.dumps()`.
6.  If a plot is requested, generate it, save it as a base64-encoded data URI string, and include it as a value in the final JSON object.
7.  Import all necessary libraries (e.g., `pandas`, `json`, `duckdb`, `matplotlib`, `seaborn`, `base64`, `io`).
8.  The script must be complete and runnable from top to bottom.

DATA SUMMARY:
{data_summary}

USER QUESTIONS:
{user_questions}

Now, write the complete Python script based on the Data Summary and User Questions.
"""

DATABASE_CODE_FIXING_PROMPT = """
The following Python script failed to execute correctly. Analyze the original question, data summary, broken code, and error message to fix the script.

CRITICAL FIXING INSTRUCTIONS:
1.  Understand the error message and the code's intent. The error was likely caused by incorrect column names, data types, or library usage.
2.  If the data source is DuckDB, ensure `httpfs` and `parquet` extensions are loaded and that date functions like `strptime` are used correctly.
3.  If the data source is a local file, ensure the file path placeholder is `'__FILE_PATH__'`.
4.  Correct the script so it is complete, runnable, and prints EXACTLY one JSON object to standard output as its final action.
5.  Return ONLY the fixed and complete Python code inside a ` ```python...``` ` block. Do not add explanations.

ORIGINAL QUESTION:
{user_questions}

DATA SUMMARY:
{data_summary}

BROKEN CODE:
```python
{code_to_fix}
ERROR MESSAGE:
{error_message}
"""

#General Question Answering Prompt for Final Formatting (Used by the web scraping workflow)
FINAL_ANSWER_SYSTEM_PROMPT = """You are a data analyst..."""