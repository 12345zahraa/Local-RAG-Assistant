from __future__ import annotations

import os
from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma2:2b"
EMBEDDING_MODEL = "nomic-embed-text"  # <-- مدل مخصوص امبدینگ اضافه شد
LOCAL_QDRANT_PATH = "./local_qdrant"
COLLECTION_NAME = "enterprise_db"


def get_llm() -> OllamaLLM:
    """Create a configured Ollama chat model instance."""
    return OllamaLLM(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
    )


def get_embeddings() -> OllamaEmbeddings:
    """Create a configured Ollama embeddings model instance."""
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,  # <-- استفاده از مدل مخصوص بردارسازی
        base_url=OLLAMA_BASE_URL,
    )


def _ensure_local_storage() -> None:
    os.makedirs(LOCAL_QDRANT_PATH, exist_ok=True)


def process_and_index_text(text: str) -> int:
    """
    Split raw text into chunks, create document objects, and persist them
    in a local Qdrant collection using direct file path storage.
    """
    if not text or not text.strip():
        raise ValueError("Text content cannot be empty.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=120,
    )
    chunks = splitter.split_text(text)
    documents = [
        Document(page_content=chunk, metadata={"source": "uploaded_text"})
        for chunk in chunks
    ]

    _ensure_local_storage()

    try:
        QdrantVectorStore.from_documents(
            documents=documents,
            embedding=get_embeddings(),
            path=LOCAL_QDRANT_PATH,  
            collection_name=COLLECTION_NAME,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to index text into Qdrant: {exc}") from exc

    return len(documents)


def get_rag_chain():
    """
    Load the persisted vector store from local storage path, construct the retriever, 
    prompt, and return a ready-to-use RAG chain.
    """
    _ensure_local_storage()

    try:
        vector_store = QdrantVectorStore.from_existing_collection(
            embedding=get_embeddings(),
            path=LOCAL_QDRANT_PATH,  
            collection_name=COLLECTION_NAME,
        )
    except Exception as exc:
        raise RuntimeError(
            "No indexed documents found. Please call /index-document first."
        ) from exc

    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    def format_docs(documents: List[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in documents)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a careful, professional assistant. "
                    "Answer using only the provided context. "
                    'If the answer is not explicitly present in the context, reply exactly: "Not found in documents".'
                ),
            ),
            (
                "human", 
                "Context:\n{context}\n\nQuestion: {question}"
            ),
        ]
    )

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | get_llm()
        | StrOutputParser()
    )

    return rag_chain