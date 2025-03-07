# RAG (Retrieval Augmented Generation) System 

## Video Tutorial (Portuguese)
This repository is part of Bix Tech "Semana de Dados" a.k.a Data Week. For a further explanation aboutwhat is RAG and this tutorial, watch the video below:

[![Data Week - RAG Tutorial](https://img.youtube.com/vi/nbXG9Xh4ZIg/0.jpg)](https://www.youtube.com/watch?v=nbXG9Xh4ZIg)

🎥 [Assista ao tutorial no YouTube](https://www.youtube.com/watch?v=nbXG9Xh4ZIg)

---

A Question-Answering system built using LangChain and ChromaDB that allows users to query their documents using natural language. The system uses OpenAI's language models to provide context-aware answers based on the content of the indexed documents.

## Overview

This system allows you to:
- Index text documents
- Create and manage a vector store using ChromaDB
- Interact with your documents through a chat interface
- Get answers with source references
- View and manage the document store


## Repository Structure

```
rag_system/
├── documents/         # Place your text files here
├── logs/             # System logs are stored here
├── db/               # Vector store database (created automatically)
├── core.py           # Core RAG system implementation
├── interface.py      # Interactive command-line interface
├── requirements.txt  # Project dependencies
└── .env             # Environment variables configuration
```

## Setup Instructions

### 1. Create a Virtual Environment

For Windows:
```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
.venv\Scripts\activate
```

For Linux/Mac:
```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the root directory with the following content:
```env
OPENAI_API_KEY=your-api-key-here
MODEL_NAME=gpt-3.5-turbo
COLLECTION_NAME=my_documents
PERSIST_DIRECTORY=db
```

Replace `your-api-key-here` with your actual OpenAI API key.

### 4. Prepare Documents

1. Create a `documents` directory if it doesn't exist:
```bash
mkdir documents
```

2. Place your text files (`.txt`) in the `documents` directory. These are the documents that will be indexed and used for answering questions.

### 5. Running the Application

```bash
# Run the interactive interface
python interface.py
```

## Using the System

Once running, the system provides the following options:

1. **Index documents**: Processes and indexes all text files in the `documents` directory
2. **Check total number of documents**: Shows how many documents are currently indexed
3. **Delete document store**: Removes all indexed documents
4. **Start RAG chat**: Begins an interactive Q&A session
5. **Exit**: Closes the application

### Chat Commands
When in chat mode:
- Type your questions normally and press Enter
- Type 'sources' to see detailed source documents for the last answer
- Type 'quit', 'exit', or 'q' to return to the main menu

## Requirements

- Python 3.8 or higher
- OpenAI API key
- Sufficient disk space for document storage
- Internet connection for API access

## Dependencies

Main libraries used:
- langchain
- langchain-openai
- langchain-community
- langchain-chroma
- chromadb
- python-dotenv
- rich

## Troubleshooting

If you encounter any issues:

1. Check the logs in the `logs` directory
2. Ensure your OpenAI API key is valid
3. Verify that your documents are text files (.txt)
4. Make sure all required directories exist
5. Check your internet connection

## Additional Notes

- The system creates necessary directories automatically
- Logs are timestamped and stored in the `logs` directory
- The vector store is persistent and stored in the `db` directory
- All text files should be in UTF-8 or compatible encoding

For any other issues or questions, please refer to the logs or create an issue in the repository.