# src/core/database.py

import pandas as pd
from typing import Tuple, Optional
import psycopg2

class DatabaseManager:
    def __init__(self, connection_string: str):
        """
        Initialize DatabaseManager with PostgreSQL connection string
        
        Args:
            connection_string (str): PostgreSQL connection string
        """
        self.connection_string = connection_string
        self._validate_connection()

    def _validate_connection(self):
        """Validate that the PostgreSQL database connection is valid"""
        try:
            conn = self._get_postgres_connection()
            conn.close()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to PostgreSQL database: {str(e)}")

    def _get_postgres_connection(self):
        """Get a connection to the PostgreSQL database"""
        return psycopg2.connect(self.connection_string)

    def execute_query(self, query: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Execute an SQL query against PostgreSQL and return results as a DataFrame
        
        Args:
            query (str): SQL query to execute
            
        Returns:
            Tuple[Optional[pd.DataFrame], Optional[str]]: 
                - DataFrame with results if successful, None if failed
                - Error message if failed, None if successful
        """
        try:
            conn = self._get_postgres_connection()
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df, None
        except Exception as e:
            error_message = f"Error executing query: {str(e)}"
            return None, error_message

    def get_table_schema(self, table_name: str) -> Optional[str]:
        """
        Get the schema of a specific PostgreSQL table
        
        Args:
            table_name (str): Name of the table
            
        Returns:
            Optional[str]: Table schema if successful, None if failed
        """
        try:
            conn = self._get_postgres_connection()
            cursor = conn.cursor()
            
            # Get table schema information
            schema_query = """
            SELECT 
                column_name, 
                data_type,
                is_nullable
            FROM 
                information_schema.columns 
            WHERE 
                table_name = %s
            ORDER BY 
                ordinal_position;
            """
            cursor.execute(schema_query, (table_name,))
            columns = cursor.fetchall()
            
            # Format schema information
            schema = f"CREATE TABLE {table_name} (\n"
            for i, (column_name, data_type, is_nullable) in enumerate(columns):
                nullable = "NULL" if is_nullable == "YES" else "NOT NULL"
                schema += f"    {column_name} {data_type} {nullable}"
                if i < len(columns) - 1:
                    schema += ",\n"
                else:
                    schema += "\n"
            schema += ");"
            
            conn.close()
            return schema
        except Exception as e:
            print(f"Error getting table schema: {e}")
            return None
            
    def list_tables(self) -> list:
        """
        List all tables in the public schema of the PostgreSQL database
        
        Returns:
            list: List of table names
        """
        try:
            conn = self._get_postgres_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
            tables = [row[0] for row in cursor.fetchall()]
            conn.close()
            return tables
        except Exception as e:
            print(f"Error listing tables: {e}")
            return []
