from __future__ import annotations

import logging
import os
from typing import List

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma2:2b"
EMBEDDING_MODEL = "nomic-embed-text"
LOCAL_QDRANT_PATH = "./local_qdrant"
COLLECTION_NAME = "enterprise_db"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K_RERANKED = 3  # Number of top documents to keep after reranking

# Store chat history per session in memory.
SESSION_STORE = {}

# Cache the vector store so we do not reopen the database for every request.
_VECTOR_STORE: QdrantVectorStore | None = None

# Cache the reranker model to avoid reloading on every request.
_RERANKER: CrossEncoder | None = None


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
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def get_reranker() -> CrossEncoder:
    """
    Return a cached CrossEncoder reranker model, initializing it only once.
    This prevents expensive model loading on every API request.
    """
    global _RERANKER

    if _RERANKER is None:
        logger.info(f"Loading reranker model: {RERANKER_MODEL_NAME}")
        try:
            _RERANKER = CrossEncoder(RERANKER_MODEL_NAME)
            logger.info("Reranker model loaded successfully")
        except Exception as exc:
            logger.error(f"Failed to load reranker model: {exc}")
            raise RuntimeError(f"Failed to load reranker model {RERANKER_MODEL_NAME}: {exc}") from exc

    return _RERANKER


def rerank_documents(documents: List[Document], query: str, top_k: int = TOP_K_RERANKED) -> List[Document]:
    """
    Rerank retrieved documents based on relevance to the query using CrossEncoder.
    
    Args:
        documents: List of documents retrieved from vector database
        query: User's query string
        top_k: Number of top-ranked documents to return
    
    Returns:
        List of top-k reranked documents sorted by relevance score
    """
    if not documents:
        return []
    
    if len(documents) <= top_k:
        # No need to rerank if we have fewer documents than top_k
        logger.debug(f"Returning all {len(documents)} documents (less than top_k={top_k})")
        return documents
    
    try:
        reranker = get_reranker()
        
        # Prepare pairs of (query, document_content) for the reranker
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Get reranking scores
        scores = reranker.predict(pairs)
        
        # Create a list of (document, score) tuples
        doc_score_pairs = list(zip(documents, scores))
        
        # Sort by score in descending order and take top_k
        ranked_docs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)[:top_k]
        
        # Extract documents and log reranking info
        reranked_docs = [doc for doc, score in ranked_docs]
        logger.debug(f"Reranked {len(documents)} documents → top {len(reranked_docs)} selected")
        
        for idx, (doc, score) in enumerate(ranked_docs, 1):
            logger.debug(f"  Rank {idx}: score={score:.4f}, content_preview={doc.page_content[:50]}...")
        
        return reranked_docs
    
    except Exception as exc:
        logger.error(f"Error during document reranking: {exc}")
        # Fallback: return original documents on error
        logger.warning("Falling back to original retrieved documents due to reranking error")
        return documents[:top_k]


def _ensure_local_storage() -> None:
    os.makedirs(LOCAL_QDRANT_PATH, exist_ok=True)


def get_vector_store() -> QdrantVectorStore:
    """Return a cached Qdrant vector store, creating it only once."""
    global _VECTOR_STORE

    if _VECTOR_STORE is None:
        _ensure_local_storage()
        try:
            _VECTOR_STORE = QdrantVectorStore.from_existing_collection(
                embedding=get_embeddings(),
                path=LOCAL_QDRANT_PATH,
                collection_name=COLLECTION_NAME,
            )
        except Exception as exc:
            raise RuntimeError(
                "No indexed documents found. Please call /index-document first."
            ) from exc

    return _VECTOR_STORE


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

    global _VECTOR_STORE

    try:
        if _VECTOR_STORE is None:
            _VECTOR_STORE = QdrantVectorStore.from_documents(
                documents=documents,
                embedding=get_embeddings(),
                path=LOCAL_QDRANT_PATH,
                collection_name=COLLECTION_NAME,
            )
        else:
            # Reuse the already initialized store to avoid reopening the DB.
            _VECTOR_STORE = get_vector_store()
            _VECTOR_STORE.add_documents(documents)
    except Exception as exc:
        raise RuntimeError(f"Failed to index text into Qdrant: {exc}") from exc

    return len(documents)


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Return a session-specific chat history object."""
    if session_id not in SESSION_STORE:
        SESSION_STORE[session_id] = InMemoryChatMessageHistory()
    return SESSION_STORE[session_id]


def get_rag_chain():
    """
    Load the persisted vector store from local storage path, construct the retriever,
    apply reranking to improve context quality, and return a ready-to-use RAG chain
    with message history support.
    """
    vector_store = get_vector_store()
    # Retrieve more documents than we'll ultimately use (we'll rerank and select top-k)
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})

    def format_docs(documents: List[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in documents)

    def retrieve_and_rerank(state: dict) -> List[Document]:
        """Retrieve documents and rerank them for improved quality."""
        question = state["question"]
        retrieved_docs = retriever.invoke(question)
        reranked_docs = rerank_documents(retrieved_docs, question, top_k=TOP_K_RERANKED)
        return reranked_docs

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
            MessagesPlaceholder(variable_name="history"),
            (
                "human",
                "Context:\n{context}\n\nQuestion: {question}",
            ),
        ]
    )

    base_chain = (
        RunnablePassthrough.assign(
            context=lambda state: format_docs(
                retrieve_and_rerank(state)
            )
        )
        | prompt
        | get_llm()
        | StrOutputParser()
    )

    return RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )