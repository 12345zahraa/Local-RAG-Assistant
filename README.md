# Local-RAG-Assistant

**Local-RAG-Assistant** is a privacy-focused Retrieval-Augmented Generation (RAG) application designed to run entirely on your local machine. It combines **FastAPI**, **Ollama**, and **reranking** to provide accurate, context-aware answers from your documents without relying on external cloud services.

## Key Features

- **Reranking**: Improves retrieval quality by re-ordering the most relevant document chunks before generation.
- **FastAPI Backend**: Provides a fast, scalable, and easy-to-use API for document processing and querying.
- **Contextual Memory**: Supports multi-turn conversations by keeping track of conversation context.
- **Local-First Privacy**: Keeps document processing and inference on your machine for better data privacy.
- **Multilingual Support**: Handles queries and content across multiple languages.

## Tech Stack

- **Backend**: FastAPI
- **LLM Runtime**: Ollama
- **RAG Workflow**: LangChain
- **Vector Database**: Qdrant
- **Reranking**: Cross-Encoder / Sentence-Transformers

## How It Works

1. **Document Ingestion**: PDFs or text files are uploaded and processed into chunks.
2. **Embedding & Storage**: Chunks are converted into embeddings and stored in a local vector database.
3. **Retrieval**: The system searches for the most relevant chunks based on the user query.
4. **Reranking**: Retrieved chunks are reranked to improve relevance and accuracy.
5. **Answer Generation**: The best context is sent to the LLM to generate a precise response.

## Installation & Usage

Clone the repository:

```bash
git clone https://github.com/12345zahraa/Local-RAG-Assistant.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

### Run the application:

To start the interface, run the following command:
```bash
streamlit run app_ui.py
