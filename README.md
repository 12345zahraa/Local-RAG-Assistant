Local RAG Assistant
A production-style local Retrieval-Augmented Generation (RAG) pipeline designed to provide accurate, context-aware answers from your local documents. This system leverages FastAPI, Ollama (for local LLM inference), and a Reranker model to ensure high-quality retrieval and response generation.

🚀 Key Features
High-Precision Reranking: Uses a Cross-Encoder reranker to ensure only the most relevant document chunks are processed by the LLM.

FastAPI Backend: A robust, asynchronous API architecture for document indexing and querying.

Contextual Memory: Supports session-based conversation history, allowing for natural, multi-turn interactions.

Local-First Privacy: Runs entirely on your machine using Ollama and local vector stores, ensuring your data never leaves your device.

Multilingual Support: Capable of processing documents and answering queries across different languages.

🛠 Tech Stack
Framework: FastAPI

LLM Engine: Ollama (Local Inference)

Orchestration: LangChain

Vector Store: Qdrant

Reranking: Cross-Encoder (Sentence-Transformers)

📋 How It Works
Indexing: Upload your PDF documents. The system extracts text, chunks it, and stores the vector embeddings in the local database.

Retrieval: When you ask a question, the system searches the database for relevant content.

Reranking: The retrieved chunks are evaluated and re-ordered by the Reranker model to guarantee semantic alignment with your query.

Generation: The top-ranked context is passed to the LLM to generate a precise, context-aware answer.

⚙️ Installation & Usage
Clone the repository:

Bash
git clone https://github.com/12345zahraa/Local-RAG-Assistant.git
Install requirements:

Bash
pip install -r requirements.txt
Run the API:

Bash
python main.py