import streamlit as st
import cohere
import os
from dotenv import load_dotenv
from langchain_cohere.embeddings import CohereEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from sqlalchemy import create_engine, inspect
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
import random

load_dotenv()

def get_engine_for_postgresql_db():
    """Create engine for PostgreSQL database."""
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")

    # PostgreSQL connection URL
    connection_url = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    return create_engine(connection_url)


engine = get_engine_for_postgresql_db()
db = SQLDatabase(engine)
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
cohere_api_key = os.getenv('COHERE_API_KEY')

embeddings = CohereEmbeddings(model="embed-english-light-v3.0", cohere_api_key=cohere_api_key)
parser = StrOutputParser()


# Fetch all table data from PostgreSQL
def fetch_all_table_data(engine):
    inspector = inspect(engine) 
    all_data = {}

    # Loop through all tables
    for table_name in inspector.get_table_names():
        query = f"SELECT * FROM {table_name};"
        # Execute the query and fetch data using SQLDatabase
        table_data = db.run(query)
        all_data[table_name] = table_data


        # all_data = {
        #     'contactpersons': 'data',
        #     'customers': 'data'
        # }

    return all_data

# Ingest data into Qdrant
def ingest_database_data(engine):
    # Fetch data from all tables
    all_tables_data = fetch_all_table_data(engine)

    # Text splitter for breaking the data into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=300)
    collection_name = "sql"

    try:
        qdrant_client.get_collection(collection_name)
        st.write(f"Collection {collection_name} exists.")
    except UnexpectedResponse:
        st.write(f"Collection {collection_name} does not exist. Creating collection...")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
        )
        st.write(f"Collection {collection_name} created successfully.")

    vectorstore = QdrantVectorStore(
        embedding=embeddings,
        client=qdrant_client,
        collection_name=collection_name
    )

    # Split the data into chunks
    chunks = splitter.split_text(str(all_tables_data))

    # Add the chunks to Qdrant vector store
    vectorstore.add_texts(chunks)
    st.write(f"Ingested data into Qdrant successfully.")

# Query data from Qdrant and use it for generating responses
def query_document(query):
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    vectorstore = QdrantVectorStore(
        embedding=embeddings,
        client=qdrant_client,
        collection_name="sql"
    )

    retriever = vectorstore.as_retriever()
    similar_chunks = retriever.invoke(query)

    # Combine the content of retrieved chunks
    context = " ".join([chunk.page_content for chunk in similar_chunks])

    # Initialize the cohere client with the provided API key
    co = cohere.Client(api_key=cohere_api_key)

    # Send the message and get the response
    response = co.chat(
        message=f"Context: {context}\n\nQuery: {query}\n\nAnswer"
    )

    text = response.text
    #st.write(text)

    # Return the parsed response
    return parser.parse(text)