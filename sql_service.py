# src/services/sql_service.py

from typing import Optional, Tuple
import pandas as pd
# Use relative imports
from model import get_model_instance
from database import DatabaseManager
from prompts import get_sql_prompt
from helpers import clean_sql_response, validate_sql_syntax  # Import new helpers

class SQLService:
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize SQLService with database manager
        
        Args:
            db_manager (DatabaseManager): Instance of DatabaseManager
        """
        self.db_manager = db_manager
        self.model = get_model_instance()

    def generate_sql_query(
        self, 
        user_query: str, 
        table_statement: str,
        previous_query: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
        """
        Generate and execute SQL query based on user input
        
        Args:
            user_query (str): User's natural language query
            table_statement (str): Database table schema
            previous_query (Optional[str]): Previous failed query
            error_message (Optional[str]): Previous error message
            
        Returns:
            Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
                - DataFrame with results if successful, None if failed
                - Generated SQL query
                - Error message if failed, None if successful
        """
        max_attempts = 3
        attempts = 0
        
        db_type = "PostgreSQL"
        current_sql_query = previous_query  # Keep track of the latest generated query
        
        while attempts < max_attempts:
            # Generate prompt using the new function
            prompt = get_sql_prompt(
                user_query,
                table_statement,
                current_sql_query,  # Pass the most recent query attempt
                error_message,
                db_type=db_type
            )
            
            # Get response from model
            sql_response = self.model.generate_response(prompt)
            
            # Clean up response
            sql_query = clean_sql_response(sql_response)
            current_sql_query = sql_query  # Update the latest generated query
            
            # --- Basic Syntax Validation ---
            is_valid, validation_error = validate_sql_syntax(sql_query)
            if not is_valid:
                error_message = f"Generated query failed syntax validation: {validation_error}. Query: {sql_query}"
                attempts += 1
                # Optionally, provide specific feedback to the LLM about the syntax error
                # For now, we just retry with the validation error message.
                continue  # Skip execution and retry generation
            # --- End Validation ---
            
            # Execute query
            df, db_error = self.db_manager.execute_query(sql_query)
            
            if df is not None:
                return df, sql_query, None  # Success
                
            # Prepare for next attempt
            error_message = f"Database execution error: {db_error}"  # More specific error
            attempts += 1
            
        # Failed after all attempts
        final_error = f"Failed to generate and execute a valid SQL query after {max_attempts} attempts. Last generated query: '{current_sql_query}'. Last error: {error_message}"
        return None, current_sql_query, final_error
