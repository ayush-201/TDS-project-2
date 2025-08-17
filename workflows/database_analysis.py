import logging
import json
import re
import subprocess
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from core.base import BaseWorkflow
from utils.prompts import (
    DATABASE_CODE_GENERATION_SYSTEM_PROMPT,
    DATABASE_CODE_FIXING_PROMPT,
)

logger = logging.getLogger(__name__)

# --- Helper Functions ---
def make_json_serializable(obj):
    """IMPROVED: Recursively convert pandas/numpy objects to JSON-serializable formats."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return None if np.isnan(obj) or np.isinf(obj) else float(obj)
    if isinstance(obj, (pd.Series, np.ndarray)):
        return [make_json_serializable(item) for item in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    return obj

def extract_json_from_output(output: str) -> str:
    """Extracts the first valid JSON object or array from a string."""
    output = output.strip()
    match = re.search(r'(\{.*\}|\[.*\])', output, re.DOTALL)
    if match:
        return match.group(0)
    logger.warning("Could not find a valid JSON object or array in the script's output.")
    return output

# --- Main Workflow Class ---
class DatabaseAnalysisWorkflow(BaseWorkflow):
    """
    A general-purpose workflow that analyzes data from either a local file
    or a remote source described in the prompt, then generates and executes
    Python code to answer the user's questions.
    """
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        task_description = input_data.get("task_description", "")
        file_path = input_data.get("file_path") # This will be None if no file is uploaded

        logger.info(f"Starting database analysis workflow. Local file provided: {file_path is not None}")

        data_summary = self._create_data_summary(task_description, file_path)

        generated_code = await self._generate_python_code(task_description, data_summary)

        execution_result = await self._execute_and_fix_code(generated_code, task_description, data_summary)

        try:
            json_output_str = extract_json_from_output(execution_result)
            final_result = json.loads(json_output_str)
            return make_json_serializable(final_result)
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Final output was not valid JSON. Error: {e}\nOutput:\n{execution_result}")
            raise ValueError("The script's final output was not valid JSON.")

    def _create_data_summary(self, task_description: str, file_path: Optional[str]) -> str:
        """
        Creates a data summary. If a local file is provided, it reads the file header.
        Otherwise, it extracts the schema from the task description text.
        """
        if file_path:
            # --- Case 1: A local file was uploaded ---
            logger.info(f"Creating data summary from local file: {file_path}")
            try:
                if file_path.lower().endswith('.csv'):
                    df_head = pd.read_csv(file_path, nrows=5)
                elif file_path.lower().endswith(('.xls', '.xlsx')):
                    df_head = pd.read_excel(file_path, nrows=5)
                else:
                    raise ValueError(f"Unsupported file type: {file_path}")
                
                schema = df_head.dtypes.to_string()
                # The placeholder '__FILE_PATH__' will be used in the prompt
                return f"Data Source: Local file\nFile Path Placeholder: '__FILE_PATH__'\n\nSchema:\n{schema}\n\nFirst 5 rows:\n{df_head.to_string()}"
            except Exception as e:
                raise ValueError(f"Could not read the provided data file at {file_path}. Error: {e}")
        else:
            # --- Case 2: No local file, extract info from the prompt ---
            logger.info("Creating data summary from the task description text.")
            s3_match = re.search(r"s3://[^\s`]+", task_description)
            schema_match = re.search(r"Here are the columns in the data:([\s\S]*)", task_description)
            
            if s3_match and schema_match:
                s3_path = s3_match.group(0)
                schema_text = schema_match.group(1)
                return f"Data Source: Remote DuckDB query\nS3 Path: {s3_path}\n\nSchema Information:\n{schema_text.strip()}"
            else:
                raise ValueError("The request does not contain a local data file or a valid data source description (S3 path and schema) in the text.")

    async def _generate_python_code(self, task: str, summary: str, code_to_fix: str = "", error: str = "") -> str:
        """Generates or fixes Python code using the LLM."""
        if code_to_fix:
            prompt_template = DATABASE_CODE_FIXING_PROMPT
            prompt_input = {
                "user_questions": task, "data_summary": summary,
                "code_to_fix": code_to_fix, "error_message": error
            }
        else:
            prompt_template = DATABASE_CODE_GENERATION_SYSTEM_PROMPT
            prompt_input = {"data_summary": summary, "user_questions": task}
        
        full_prompt = prompt_template.format(**prompt_input)
        response_obj = await self.llm.ainvoke(full_prompt)
        response_str = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)
        
        match = re.search(r'```python\n(.*?)```', response_str, re.DOTALL)
        if match:
            return match.group(1).strip()
        logger.warning("LLM response did not contain a valid python code block, returning raw response.")
        return response_str.strip()

    async def _execute_and_fix_code(self, code: str, task: str, summary:str, max_retries: int = 1) -> str:
        """Executes the Python script and attempts to fix it if it fails."""
        current_code = code
        for attempt in range(max_retries + 1):
            logger.info(f"Executing generated code (Attempt {attempt + 1}/{max_retries + 1})")
            
            # The generated code should now be self-contained and not need file path replacement
            with open("temp_database_generated_code.py", "w", encoding="utf-8") as f:
                f.write(current_code)
            
            try:
                result = subprocess.run(
                    ["python", "-W", "ignore", "temp_database_generated_code.py"], # Added -W ignore to suppress warnings
                    capture_output=True, text=True, timeout=300, check=True
                )
                logger.info("Code executed successfully.")
                return result.stdout
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                error_output = e.stderr if hasattr(e, 'stderr') else "Execution timed out."
                logger.warning(f"Code execution failed on attempt {attempt + 1}. Error:\n{error_output}")
                
                if attempt < max_retries:
                    logger.info("Attempting to fix the code...")
                    current_code = await self._generate_python_code(task, summary, code_to_fix=current_code, error=error_output)
                else:
                    raise ValueError(f"Code failed after {max_retries + 1} attempts. Last error: {error_output}")
        raise RuntimeError("Exited execution loop unexpectedly.")