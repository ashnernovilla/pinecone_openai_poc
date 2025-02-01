# Pinecone Vector Store and LlamaIndex PoC

## Overview
This project is a Proof of Concept (PoC) for integrating Pinecone vector storage with LlamaIndex, using a Streamlit-based user interface. The application allows users to upload text files, store them in Pinecone as vector embeddings, and query them through LlamaIndex.

## Features
- **API Key Management**: Users can input their Pinecone and OpenAI API keys via the Streamlit sidebar.
- **Document Upload**: Uploads and processes text files for indexing.
- **Pinecone Initialization**: Creates and manages a Pinecone index for storing vector embeddings.
- **Vector Store Indexing**: Converts uploaded documents into vector embeddings and stores them in Pinecone.
- **Query Interface**: Allows users to ask questions and retrieve relevant responses using LlamaIndex.

## Requirements
Ensure you have the following installed before running the application:
- Python 3.8+
- Streamlit
- Pinecone SDK
- LlamaIndex
- OpenAI API access

## Installation
1. Clone the repository:
   ```sh
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
1. Run the Streamlit application:
   ```sh
   streamlit run app.py
   ```
2. Enter your Pinecone and OpenAI API keys in the sidebar.
3. Upload a text file to process.
4. Initialize the Pinecone index and build the vector store.
5. Enter a query in the chat interface to retrieve relevant responses.

## Environment Variables
Set the following environment variables if not using Streamlit’s input fields:
```sh
export PINECONE_API_KEY="your_pinecone_api_key"
export OPENAI_API_KEY="your_openai_api_key"
```

## Troubleshooting
- Ensure your Pinecone and OpenAI API keys are valid.
- Check that the Pinecone index has been successfully created.
- If encountering document upload issues, verify file format compatibility.

## Link
