# src/utils/prompts.py

from typing import Optional

def get_sql_prompt(
    user_query: str,
    table_statement: str,
    previous_query: Optional[str] = None,
    error_message: Optional[str] = None,
    db_type: str = "PostgreSQL"
) -> str:
    """
    Generate a detailed prompt for the LLM to create a SQL query.

    Args:
        user_query (str): The user's natural language query.
        table_statement (str): The CREATE TABLE statement for the relevant table.
        previous_query (Optional[str]): The previously attempted SQL query, if any.
        error_message (Optional[str]): The error message from the previous attempt, if any.
        db_type (str): The specific SQL dialect (e.g., "PostgreSQL").

    Returns:
        str: The generated prompt string.
    """
    prompt = f"""You are an expert {db_type} data analyst. Your task is to generate a {db_type} SQL query based on the user's question and the provided table schema.

Database Schema:
```sql
{table_statement}
```

User Question: "{user_query}"

Instructions:
1.  Analyze the user question and the table schema carefully.
2.  Generate a SINGLE, syntactically correct {db_type} query that answers the user's question.
3.  ONLY output the SQL query. Do not include any explanations, comments, or markdown formatting (like ```sql).
4.  Ensure the query is compatible with standard {db_type} syntax.
5.  If the question cannot be answered with the given schema, respond with "Error: Cannot answer question with the provided schema."
"""

    if previous_query and error_message:
        prompt += f"""
Correction Attempt:
The previous query attempt failed.
Previous Query:
```sql
{previous_query}
```
Error Message: "{error_message}"

Please analyze the error message and the previous query, then generate a corrected {db_type} SQL query based on the original user question and schema. ONLY output the corrected SQL query.
"""
    else:
        prompt += """
Generate the {db_type} SQL query now:
"""

    return prompt

def get_analysis_prompt(
    user_input: str,
    table_info: str,
    previous_code: Optional[str] = None,
    execution_error: Optional[str] = None
) -> str:
    """
    Generate analysis code prompt
    
    Args:
        user_input (str): User's natural language query
        table_info (str): Sample data information
        previous_code (Optional[str]): Previous failed code
        execution_error (Optional[str]): Previous error message
        
    Returns:
        str: Generated prompt
    """
    return f"""
You are a data analysis assistant. Use the table information below to generate Python code that performs the analysis for the user's request. The code should use the provided dataframe 'df' and compute the main result, assigning it to a variable named 'result' which is returned at the end. Do not use print statements. Only output the Python code.
Generate the code such that the results should be shown on the streamlit app, like plotted graphs should be shown using "st.pyplot()" or "st.image()" functions.

USER INPUT: {user_input}

TABLE INFORMATION:
{table_info}

{f"Below is the information regarding the previous code generated and error when executed, modify the code to fix the error while maintaining the original intent." if execution_error else ""}
{f"Execution Error: {execution_error}" if execution_error else ""}
{f"Previous Python Code: {previous_code}" if previous_code else ""}

<python_code>
"""
