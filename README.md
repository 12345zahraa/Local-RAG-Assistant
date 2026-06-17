# Local RAG Assistant

A lightweight local Retrieval-Augmented Generation (RAG) pipeline that lets you upload PDF documents, index their content, and ask questions through a Streamlit frontend.

## Features
- FastAPI backend for indexing and querying documents
- Streamlit frontend for chat-based interaction
- PDF upload support for indexing
- Local vector storage for retrieval

## Requirements
- Python 3.10+
- Ollama running locally with the required models
- `pip` or `uv` for dependency management

## Setup

### 1. Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
.\.venv\Scripts\activate       # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> If you do not yet have a `requirements.txt` file, install the main packages manually:
>
> ```bash
> pip install fastapi uvicorn streamlit requests pypdf python-dotenv
> ```

### 3. Run the backend
Start the FastAPI server from the project root:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

The API will be available at:
- `http://127.0.0.1:8000/docs`

### 4. Run the frontend
In a separate terminal, start the Streamlit app:

```bash
streamlit run app_ui.py
```

The frontend will open in your browser.

## API Endpoints
- `POST /index-document` — Upload a PDF and index its content
- `GET /ask?question=...` — Ask a question using the local RAG pipeline

## Notes
- Make sure the backend server is running before using the frontend.
- If your local model is slow, allow extra time for responses.
