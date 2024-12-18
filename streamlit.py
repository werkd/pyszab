import logging
import os
import random

import streamlit as st
import cohere

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from sqlalchemy import create_engine, inspect
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.output_parsers import StrOutputParser

from helper_functions import query_document, ingest_database_data, fetch_all_table_data, engine

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configure page layout
st.set_page_config(
    page_title="EzQuery.ai",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()

# Enhanced custom CSS with dark theme support
st.markdown("""
<style>
    /* Theme colors */
    :root {
        --primary-color: #1E88E5;
        --background-color: #0E1117;
        --text-color: #E0E0E0;
        --card-background: #1E1E1E;
        --border-color: #2D2D2D;
    }

    .main-header {
        font-size: 2.8rem;
        color: var(--primary-color);
        margin-bottom: 1.5rem;
        font-weight: 600;
    }
    
    .sub-header {
        font-size: 1.8rem;
        color: var(--text-color);
        margin-bottom: 2rem;
        opacity: 0.9;
    }
    
    .info-box {
        padding: 1.5rem;
        border-radius: 0.75rem;
        background-color: var(--card-background);
        border: 1px solid var(--border-color);
        margin-bottom: 1rem;
        color: var(--text-color);
    }
    
    .result-container {
        background-color: var(--card-background);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid var(--border-color);
        margin-top: 1rem;
    }
    
    /* Accessibility improvements */
    .stButton button {
        min-height: 44px;
        font-size: 1rem;
        transition: all 0.2s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        .sub-header {
            font-size: 1.4rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Main layout with improved structure
st.markdown("<h1 class='main-header'>EzQuery.ai</h1>", unsafe_allow_html=True)

# Sidebar improvements
with st.sidebar:
    st.markdown("### Database Actions")
    
    with st.expander("ℹ️ Getting Started", expanded=True):
        st.markdown("""
        ### How to use EzQuery.ai:
        1. **Connect Database**
           - Click 'Ingest Data' below
           - Wait for confirmation
        2. **Query Your Data**
           - Type your question naturally
           - Use the examples for guidance
        3. **Explore Results**
           - View structured responses
           - Export data if needed
        """)

    ingest_btn = st.button(
        "🔄 Ingest Data",
        help="Load your PostgreSQL data into the system",
        use_container_width=True
    )

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Ask about your data")
    
    # Query input with examples
    example_queries = [
        "Show me all active projects",
        "How many employees are there?",
        "List top customers by revenue"
    ]
    
    # Initialize session state for user query if not already done
    if 'user_query' not in st.session_state:
        st.session_state.user_query = ""

    # User input for query
    user_query = st.text_input(
        "Enter your question in natural language:",
        placeholder=random.choice(example_queries),
        help="Try asking about projects, employees, or customers",
        value=st.session_state.user_query  # Set the value from session state
    )
    
    # Action buttons
    col_query, col_clear = st.columns([4, 1])
    with col_query:
        query_button = st.button("🔍 Get Answer", use_container_width=True)
    with col_clear:
        clear_button = st.button("🗑️ Clear", use_container_width=True)

    # Results display
    if query_button and user_query:
        st.session_state.user_query = user_query  # Store the user query in session state
        st.info("Results are being displayed")
        with st.spinner("Analyzing your query..."):
            try:
                logger.debug(f"User Query: {user_query}")  # Log user query
                response = query_document(user_query)
                logger.debug(f"Response: {response}")  # Log response
                
                # Check if response is valid
                if response:
                    st.markdown("### Results")
                    with st.container():
                        st.markdown(f"""
                        <div class='result-container'>
                            <div style='color: var(--text-color);'>
                                {response}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Add export option
                        st.download_button(
                            "📥 Export Results",
                            response,
                            file_name="query_results.txt",
                            mime="text/plain"
                        )
                else:
                    st.warning("No results found for your query.")
                    
            except Exception as e:
                logger.error(f"Error: {str(e)}")  # Log the error
                st.error(f"❌ Error: {str(e)}\nPlease try rephrasing your question.")

    # Clear button functionality
    if clear_button:
        st.session_state.user_query = ""  # Clear the session state
        st.experimental_rerun()  # Rerun the app to refresh the UI

with col2:
    # Database stats and info
    with st.container():
        st.markdown("### Database Overview")
        st.markdown("""
        <div class='info-box'>
            • Connected to: PostgreSQL
            • Tables Available: 7
            • Last Updated: Just now
        </div>
        """, unsafe_allow_html=True)

# Footer with improved layout
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    st.markdown("Made with ❤️ by EzQuery.ai Team")

cohere_api_key = os.getenv("COHERE_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
# print(qdrant_url, 'urls')
# QDRANT_URL = 'https://c07c8e76-8d02-4261-9644-e3a0f6e43acf.eu-central-1-0.aws.cloud.qdrant.io:6333/'

if not cohere_api_key or not qdrant_url or not qdrant_api_key:
    st.error("Error: Missing API keys or Qdrant URL. Please check your environment variables.")
    st.stop()



st.sidebar.title("Actions")

if st.sidebar.button("Ingest Data into Qdrant"):
    try:
        ingest_database_data(engine)
        st.success("Data ingestion completed.")
    except Exception as e:
        st.error(f"Error during ingestion: {e}")

# Query input for Qdrant
user_query = st.text_input("Enter your query:", "")
if st.button("Submit Query"):
    if user_query:
        try:
            logger.debug(f"Qdrant User Query: {user_query}")
            response = query_document(user_query)
            logger.debug(f"Qdrant Response: {response}")
            st.write(f"Response: {response}")
        except Exception as e:
            logger.error(f"Error during Qdrant query: {e}")
            st.error(f"Error during query: {e}")
    else:
        st.warning("Please enter a query.")
