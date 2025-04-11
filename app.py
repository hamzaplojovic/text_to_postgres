# src/app.py

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from database import DatabaseManager
from sql_service import SQLService
from analysis_service import AnalysisService

# Load environment variables
load_dotenv()

def init_session_state():
    """Initialize session state variables"""
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = None
    if 'sql_service' not in st.session_state:
        st.session_state.sql_service = None
    if 'analysis_service' not in st.session_state:
        st.session_state.analysis_service = None
    if 'selected_table' not in st.session_state:
        st.session_state.selected_table = None

def setup_services(connection_string):
    """Set up database and analysis services for PostgreSQL"""
    try:
        from model import get_model_instance
        get_model_instance()

        db_manager = DatabaseManager(connection_string)
        st.session_state.db_manager = db_manager
        st.session_state.sql_service = SQLService(db_manager)
        st.session_state.analysis_service = AnalysisService()
        return True
    except Exception as e:
        st.error(f"Error setting up services: {type(e).__name__}: {str(e)}")
        import traceback
        st.text(traceback.format_exc())
        return False

def main():
    st.title("PostgreSQL Data Analysis Copilot 🤖")
    
    init_session_state()
    
    st.sidebar.header("PostgreSQL Database Connection")
    
    conn_string = st.sidebar.text_input(
        "PostgreSQL Connection String", 
        placeholder="postgresql://user:password@host:port/database"
    )
    
    if conn_string:
        if 'db_manager' not in st.session_state or st.session_state.db_manager is None or st.session_state.db_manager.connection_string != conn_string:
            if setup_services(conn_string):
                st.sidebar.success("Database connected successfully!")
            else:
                st.stop()

        if st.session_state.db_manager:
            tables = st.session_state.db_manager.list_tables()
            if tables:
                st.session_state.selected_table = st.sidebar.selectbox(
                    "Select Table", tables, key=f"table_select_{conn_string}"
                )
                
                if st.session_state.selected_table:
                    table_schema = st.session_state.db_manager.get_table_schema(st.session_state.selected_table)
                    if table_schema:
                        st.sidebar.text("Table Schema:")
                        st.sidebar.code(table_schema)
                        display_analysis_interface(table_schema)
                    else:
                        st.sidebar.warning(f"Could not retrieve schema for table {st.session_state.selected_table}.")
                else:
                    st.sidebar.info("Select a table to proceed.")
            else:
                st.sidebar.warning("No tables found in the 'public' schema or failed to list tables.")
    else:
        st.sidebar.info("Please enter your PostgreSQL connection string.")

def display_analysis_interface(table_schema):
    """Display the analysis interface"""
    st.header("Ask Your Question")
    user_query = st.text_area("What would you like to analyze?", 
                            placeholder="e.g., Show me the average income by city")
    st.caption("Note: The AI tries its best, but generated SQL and analysis may require verification.")

    if st.button("Analyze"):
        if user_query:
            if not st.session_state.sql_service or not st.session_state.analysis_service:
                st.error("Services not initialized. Please check connection and setup.")
                st.stop()

            with st.spinner("Generating SQL query..."):
                df, sql_query, error = st.session_state.sql_service.generate_sql_query(
                    user_query, table_schema
                )
                
                if error:
                    st.error(error)
                else:
                    st.subheader("SQL Query")
                    st.code(sql_query, language="sql")
                    
                    st.subheader("Retrieved Data Sample")
                    st.dataframe(df.head())
                    
                    if df is None or df.empty:
                        st.warning("No data retrieved from SQL query. Cannot perform analysis.")
                        st.stop()

                    with st.spinner("Generating analysis..."):
                        if st.session_state.analysis_service:
                            analysis_code, analysis_error = st.session_state.analysis_service.generate_analysis(
                                user_query, df
                            )
                        else:
                            analysis_code, analysis_error = None, "Analysis service not available."

                        if analysis_error:
                            st.error(analysis_error)
                        else:
                            st.subheader("Analysis Code")
                            st.code(analysis_code, language="python")
                            
                            try:
                                local_namespace = {'pd': pd, 'df': df, 'st': st}
                                exec(analysis_code, {}, local_namespace)
                                result = local_namespace.get('result')
                                
                                st.subheader("Analysis Result")
                                if isinstance(result, pd.DataFrame):
                                    st.dataframe(result)
                                else:
                                    st.write(result)
                                    
                            except Exception as e:
                                st.error(f"Error executing or displaying analysis results: {str(e)}")
        else:
            st.warning("Please enter a question to analyze.")
    
if __name__ == "__main__":
    main()
