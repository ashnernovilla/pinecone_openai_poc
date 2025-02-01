# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 03:01:41 2025

@author: ASHNER_NOVILLA
"""

import streamlit as st
import os
import logging
from pinecone import Pinecone, ServerlessSpec
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext
st.set_page_config(layout="wide")

# Set up logging
logging.basicConfig(level=logging.INFO)

# Streamlit app title
st.markdown("<h1 style='text-align: center; color: black;'>Pinecone Vector Store and LlamaIndex PoC </h1>", unsafe_allow_html=True)


# API keys
st.sidebar.markdown(" **OPEN AI AND PINECONE API REQUIRED** ")

PINECONE_API_KEY = st.sidebar.text_input("Enter your Pinecone API Key", type="password")
OPENAI_API_KEY = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

if PINECONE_API_KEY and OPENAI_API_KEY:
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    # File upload
    st.subheader("Upload Documents")
    uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

    if uploaded_file:
        try:
            upload_dir = "uploaded_files"
            os.makedirs(upload_dir, exist_ok=True)  # Ensure directory exists
            file_path = os.path.join(upload_dir, uploaded_file.name)

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            documents = SimpleDirectoryReader(upload_dir).load_data()
            st.session_state['documents'] = documents  # Save documents in session state
            st.success("Document uploaded and loaded successfully.")
        except Exception as e:
            st.error(f"Error loading uploaded document: {e}")

    # Initialize Pinecone
    st.subheader("Pinecone Initialization")
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)

        # Delete and create a new index
        if st.button("Create Pinecone Index"):
            pc.delete_index("quickstart")
            pc.create_index(
                name="quickstart",
                dimension=1536,
                metric="euclidean",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            st.success("Pinecone index created successfully.")

        pinecone_index = pc.Index("quickstart")
    except Exception as e:
        st.error(f"Error initializing Pinecone: {e}")

    # Build vector store index
    st.subheader("Build Vector Store Index")
    if st.button("Initialize Vector Store and Index"):
        if 'documents' in st.session_state:
            try:
                vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                index = VectorStoreIndex.from_documents(
                    st.session_state['documents'], storage_context=storage_context
                )
                st.session_state['index'] = index  # Save index in session state
                st.success("Vector store and index initialized successfully.")
            except Exception as e:
                st.error(f"Error initializing vector store and index: {e}")
        else:
            st.error("Please upload documents before initializing the index.")

    # Chat interface
    st.subheader("Query Index via Chat")
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    query = st.text_input("Ask a question:")
    if st.button("Send Query") and query:
        if 'index' in st.session_state:
            try:
                query_engine = st.session_state['index'].as_query_engine()
                response = query_engine.query(query)
                st.session_state['chat_history'].append((query, str(response)))
            except Exception as e:
                st.error(f"Error querying index: {e}")
        else:
            st.error("Index has not been initialized. Please initialize the index first.")

    for question, answer in st.session_state['chat_history']:
        st.markdown(f"**You:** {question}")
        st.markdown(f"**Response:** {answer}")
