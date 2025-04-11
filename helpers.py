# src/utils/helpers.py

import re
import sqlparse
from typing import Optional, Tuple

def extract_python_code(text: str) -> str:
    """
    Extract Python code from text that may contain markdown code blocks
    
    Args:
        text (str): Text containing Python code
        
    Returns:
        str: Extracted Python code
    """
    # Try to find code between ```python and ``` markers
    pattern = r"```python\s*(.*?)\s*```"
    code_snippets = re.findall(pattern, text, re.DOTALL)
    
    if code_snippets:
        return code_snippets[0]
    
    # If no code blocks found, return the original text
    # after removing any potential ``` markers
    return text.replace("```", "").strip()

def clean_sql_response(sql_response: str) -> str:
    """
    Cleans the raw SQL response from the LLM.
    Removes markdown code blocks, leading/trailing whitespace, and semicolons.

    Args:
        sql_response (str): The raw response from the LLM.

    Returns:
        str: The cleaned SQL query.
    """
    # Remove markdown code blocks
    sql_response = re.sub(r"```sql\n?", "", sql_response)
    sql_response = re.sub(r"```", "", sql_response)

    # Remove leading/trailing whitespace and semicolons
    cleaned_query = sql_response.strip().rstrip(';')

    return cleaned_query

def validate_sql_syntax(sql_query: str) -> Tuple[bool, Optional[str]]:
    """
    Validates the basic syntax of the SQL query using sqlparse.
    Note: This is a basic check and doesn't guarantee semantic correctness
          or execution success against a specific database schema.

    Args:
        sql_query (str): The SQL query to validate.

    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    try:
        parsed = sqlparse.parse(sql_query)
        # Check if parsing resulted in any statements
        if not parsed:
            return False, "Query is empty or could not be parsed."
        # A very basic check: does it look like a SELECT statement?
        # You might want more sophisticated checks depending on expected query types.
        if parsed[0].get_type().upper() != 'SELECT':
             # Allow other DML/DDL if needed, but for analysis, SELECT is common.
             # Consider adjusting this check based on requirements.
             pass # Allow non-SELECT for now, but could warn or error here.
        # Further checks could involve looking for specific tokens or structures.
        return True, None
    except Exception as e:
        # sqlparse might raise errors on severely malformed queries
        return False, f"Syntax validation error: {str(e)}"
