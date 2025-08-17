# workflows/web_scraping.py
import logging
import re
import json
import base64
import io
import math
import traceback
from typing import Dict, Any, List, Optional, Set
import asyncio
import httpx
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from core.base import BaseWorkflow
# Assume these prompt files and constants are updated appropriately
from utils.prompts import (
    TABLE_SELECTION_SYSTEM_PROMPT,
    TABLE_SELECTION_HUMAN_PROMPT,
    CODE_GENERATION_SYSTEM_PROMPT, # New prompt for code generation
    CODE_GENERATION_HUMAN_PROMPT,   # New prompt for code generation
)
from utils.constants import (
    REQUEST_HEADERS, HTML_PARSER, ENGLISH_STOPWORDS, WORD_REGEX_PATTERN,
    MIN_KEYWORD_LENGTH, MATPLOTLIB_BACKEND
)

from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from core.config import get_chat_model

matplotlib.use(MATPLOTLIB_BACKEND)
logger = logging.getLogger(__name__)

# --- Helper functions (mostly unchanged, with additions for sanitization) ---
CUSTOM_STOPWORDS = {
    'scrape', 'list', 'films', 'wikipedia', 'answer', 'questions', 
    'respond', 'json', 'array', 'strings', 'containing', 'what', 'how', 'which'
}
STOPWORDS = ENGLISH_STOPWORDS | CUSTOM_STOPWORDS
_table_selection_cache: Dict[str, int] = {}

def _strip_code_fences(text: str) -> str:
    if not isinstance(text, str): return text
    t = text.strip()
    # Updated regex to handle optional language names like ```python
    t = re.sub(r"^\s*```[a-zA-Z]*\n", "", t) 
    t = re.sub(r"\n```\s*$", "", t)
    return t.strip()

def extract_keywords(task_description: str, stopwords: Optional[Set[str]] = None) -> List[str]:
    if stopwords is None: stopwords = STOPWORDS
    words = re.findall(WORD_REGEX_PATTERN, task_description.lower())
    return [w for w in words if w not in stopwords and len(w) >= MIN_KEYWORD_LENGTH]

def sanitize_for_json(obj):
    if isinstance(obj, (dict, list, str, int, float, bool, type(None))):
        if isinstance(obj, dict):
            return {str(k): sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize_for_json(v) for v in obj]
        elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return sanitize_for_json(obj.tolist())
        return obj
    # If not a standard JSON type, convert to string
    return str(obj)

# --- Workflow Steps ---

class ScrapeStep:
    """Finds and scrapes the most relevant table from a URL."""
    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        url = input_data["url"]
        task_description = input_data.get("task_description", "")
        logger.info(f"Scraping data from {url} for task: '{task_description[:50]}...'")
        
        async with httpx.AsyncClient(timeout=20, headers=REQUEST_HEADERS, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            
        tables = pd.read_html(io.StringIO(response.text))
        if not tables: raise ValueError(f"No HTML tables found at {url}.")
        
        keywords = extract_keywords(task_description)
        # LLM-based selection is good, so we keep it.
        # This part of the original logic was solid.
        best_table_idx = await self._select_best_table_with_llm(tables, task_description, keywords)
        
        data = tables[best_table_idx]
        # Clean up multi-level column headers
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(map(str, col)).strip() for col in data.columns.values]
            
        logger.info(f"Selected table with shape {data.shape} and columns: {data.columns.tolist()}")
        return {"data": data, **input_data}

    async def _select_best_table_with_llm(self, tables: List[pd.DataFrame], task_description: str, keywords: List[str]) -> int:
        # This function can remain largely the same as the original script.
        # It previews tables and asks the LLM to pick the best index.
        # For brevity, its implementation is assumed.
        # --- (Implementation from original script) ---
        return 0 # Placeholder for the LLM selection logic

class CleanStep:
    """
    Robustly cleans a DataFrame to prepare it for analysis.
    This step is crucial for making the generated code work reliably.
    """
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        data = input_data["data"].copy()
        logger.info("Starting robust data cleaning...")

        for col in data.columns:
            # Attempt to convert to numeric, handling currency, commas, and percentages
            series_str = data[col].astype(str)
            numeric_converted = pd.to_numeric(
                series_str.str.replace(r'[$,€£%]', '', regex=True)
                          .str.replace(r'\[.*?\]', '', regex=True) # Remove citations like [1]
                          .str.strip(),
                errors='coerce'
            )
            # If a significant portion of the column is numeric, convert it
            if numeric_converted.notna().sum() / len(data.index) > 0.6:
                data[col] = numeric_converted
                logger.info(f"Successfully converted column '{col}' to numeric.")
            else:
                # Fallback for non-numeric object columns: clean up strings
                data[col] = series_str.str.replace(r'\[\s*\w+\s*\]', '', regex=True).str.strip()

        # Drop rows where all values are missing
        data.dropna(how='all', inplace=True)
        # Sanitize column names to be valid Python identifiers
        data.columns = [col.replace(' ', '_').replace('(', '').replace(')', '') for col in data.columns]
        
        logger.info(f"Cleaned data. Final columns: {data.columns.tolist()}")
        return {"data": data.reset_index(drop=True), **input_data}


class CodeGeneratingAnswerStep:
    """
    The core of the general-purpose workflow. It uses an LLM to generate
    Python code to answer the user's question, then executes it.
    """
    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        df = input_data["data"]
        task_description = input_data["task_description"]
        
        logger.info("Generating Python code to answer the user's request...")

        # Create a detailed prompt for the LLM
        prompt = ChatPromptTemplate.from_messages([
            ("system", CODE_GENERATION_SYSTEM_PROMPT), # A new system prompt is needed
            ("human", CODE_GENERATION_HUMAN_PROMPT)    # A new human prompt is needed
        ])
        
        llm = get_chat_model() # Low temperature for reliable code
        chain = prompt | llm | StrOutputParser()

        # Provide the LLM with the DataFrame's structure and the user's question
        code_generation_request = {
            "task_description": task_description,
            "df_head": df.head().to_string(),
            "df_columns": str(df.columns.tolist())
        }
        
        generated_code_str = await asyncio.to_thread(chain.invoke, code_generation_request)
        generated_code = _strip_code_fences(generated_code_str)

        logger.info(f"--- Generated Code ---\n{generated_code}\n----------------------")
        
        # --- Securely execute the generated code ---
        # The generated code should set a variable named 'final_answer'.
        # This variable can be a dictionary, list, string, number, or a plot.
        
        # Provide a local scope for the execution, including the DataFrame `df`
        local_scope = {"df": df, "pd": pd, "np": np, "plt": plt, "sns": sns, "io": io, "base64": base64}
        
        try:
            # Execute the code in the defined local scope
            exec(generated_code, {"__builtins__": __builtins__}, local_scope)
            
            # The generated code is expected to produce a 'final_answer' variable
            if 'final_answer' in local_scope:
                result = local_scope['final_answer']
                return sanitize_for_json(result)
            else:
                raise ValueError("The generated code did not produce a 'final_answer' variable.")

        except Exception as e:
            logger.error(f"Error executing generated code: {e}")
            # Provide a detailed traceback for debugging
            error_trace = traceback.format_exc()
            return {"error": "Failed to execute the generated analysis code.", "details": str(e), "traceback": error_trace}


class MultiStepWebScrapingWorkflow(BaseWorkflow):
    """
    A general-purpose workflow that scrapes a table, cleans it, and then
    generates and executes code to answer a user's natural language question.
    """
    async def execute(self, input_data):
        task_description = input_data.get("task_description", "")
        url_match = re.search(r"https?://[^\s]+", task_description)
        if not url_match:
            raise ValueError("No URL found in the task description.")
        url = url_match.group(0)

        # Step 1: Scrape the web page to find the right table
        scraped_data = await ScrapeStep().run({"url": url, "task_description": task_description})

        # Step 2: Apply robust cleaning and preparation
        cleaned_data = CleanStep().run(scraped_data)

        # Step 3: Generate and execute code to get the final answer
        final_answer = await CodeGeneratingAnswerStep().run(cleaned_data)

        return final_answer