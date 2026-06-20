# Advanced Customization Guide - Reranking Layer

## Table of Contents
1. [Custom Reranker Strategies](#custom-reranker-strategies)
2. [Performance Optimization](#performance-optimization)
3. [Advanced Pipeline Configurations](#advanced-pipeline-configurations)
4. [Monitoring & Analytics](#monitoring--analytics)
5. [Production Deployment](#production-deployment)

---

## Custom Reranker Strategies

### Strategy 1: Weighted Reranking (Combine Multiple Scores)

Combine relevance score with document metadata:

```python
def rerank_documents_weighted(
    documents: List[Document], 
    query: str, 
    top_k: int = TOP_K_RERANKED,
    relevance_weight: float = 0.8,
    recency_weight: float = 0.2
) -> List[Document]:
    """
    Rerank using combination of relevance score and document metadata.
    Useful when you have timestamps or importance scores in metadata.
    """
    reranker = get_reranker()
    pairs = [[query, doc.page_content] for doc in documents]
    relevance_scores = reranker.predict(pairs)
    
    # Normalize metadata scores (example: using document recency)
    metadata_scores = []
    for doc in documents:
        # Example: boost recent documents
        recency_score = 1.0 if doc.metadata.get("source") == "recent" else 0.7
        metadata_scores.append(recency_score)
    
    # Combine scores
    combined_scores = [
        relevance_weight * rel_score + recency_weight * meta_score
        for rel_score, meta_score in zip(relevance_scores, metadata_scores)
    ]
    
    doc_score_pairs = list(zip(documents, combined_scores))
    ranked_docs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)[:top_k]
    
    return [doc for doc, score in ranked_docs]
```

### Strategy 2: Threshold-Based Filtering

Only keep documents above a relevance threshold:

```python
def rerank_documents_threshold(
    documents: List[Document], 
    query: str, 
    threshold: float = 0.7,
    max_k: int = TOP_K_RERANKED
) -> List[Document]:
    """
    Rerank and filter documents by relevance threshold.
    Only documents with score >= threshold are kept.
    """
    if not documents:
        return []
    
    reranker = get_reranker()
    pairs = [[query, doc.page_content] for doc in documents]
    scores = reranker.predict(pairs)
    
    doc_score_pairs = [
        (doc, score) 
        for doc, score in zip(documents, scores) 
        if score >= threshold
    ]
    
    # Sort by score and take top max_k
    ranked_docs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)[:max_k]
    
    selected = [doc for doc, score in ranked_docs]
    filtered_count = len(documents) - len(selected)
    logger.info(f"Filtered out {filtered_count} documents below threshold {threshold}")
    
    return selected
```

### Strategy 3: Diversity-Aware Reranking

Ensure selected documents cover different topics:

```python
from sklearn.preprocessing import normalize
import numpy as np

def rerank_documents_diverse(
    documents: List[Document], 
    query: str, 
    top_k: int = TOP_K_RERANKED,
    diversity_penalty: float = 0.3
) -> List[Document]:
    """
    Rerank documents with diversity penalty.
    Prevents multiple very similar documents in final selection.
    """
    reranker = get_reranker()
    pairs = [[query, doc.page_content] for doc in documents]
    relevance_scores = reranker.predict(pairs)
    
    selected = []
    selected_embeddings = []
    
    # Get embeddings for similarity calculation
    from sentence_transformers import SentenceTransformer
    embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
    doc_embeddings = embeddings_model.encode(
        [doc.page_content for doc in documents],
        convert_to_tensor=True
    )
    
    # Greedily select documents with diversity consideration
    remaining_docs = list(range(len(documents)))
    
    while len(selected) < top_k and remaining_docs:
        best_idx = None
        best_score = -float('inf')
        
        for idx in remaining_docs:
            # Base relevance score
            score = relevance_scores[idx]
            
            # Penalize similarity to already selected documents
            if selected_embeddings:
                similarities = [
                    np.dot(doc_embeddings[idx], embed)
                    for embed in selected_embeddings
                ]
                similarity_penalty = max(similarities) * diversity_penalty
                score -= similarity_penalty
            
            if score > best_score:
                best_score = score
                best_idx = idx
        
        if best_idx is not None:
            selected.append(documents[best_idx])
            selected_embeddings.append(doc_embeddings[best_idx])
            remaining_docs.remove(best_idx)
    
    logger.info(f"Selected {len(selected)} diverse documents")
    return selected
```

---

## Performance Optimization

### Strategy 1: Batch Processing for Multiple Queries

```python
from typing import Dict, List

def rerank_documents_batch(
    query_doc_pairs: List[tuple],  # List of (query, [documents]) tuples
    top_k: int = TOP_K_RERANKED
) -> Dict[str, List[Document]]:
    """
    Efficiently rerank multiple queries' documents in batch.
    Useful for concurrent requests or batch processing.
    """
    reranker = get_reranker()
    results = {}
    
    for query, documents in query_doc_pairs:
        if not documents:
            results[query] = []
            continue
        
        pairs = [[query, doc.page_content] for doc in documents]
        scores = reranker.predict(pairs)
        
        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        results[query] = [doc for doc, _ in ranked]
    
    return results
```

### Strategy 2: Caching Reranking Results

```python
from functools import lru_cache
import hashlib

class RerankerCache:
    def __init__(self, max_cache_size: int = 100):
        self.cache = {}
        self.max_size = max_cache_size
    
    def _hash_query_docs(self, query: str, doc_contents: tuple) -> str:
        """Generate hash for query + documents."""
        content = f"{query}:" + ":".join(doc_contents)
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_reranked(
        self,
        documents: List[Document],
        query: str,
        top_k: int = TOP_K_RERANKED
    ) -> List[Document]:
        """Get reranked documents with caching."""
        doc_contents = tuple(doc.page_content[:100] for doc in documents)
        cache_key = self._hash_query_docs(query, doc_contents)
        
        if cache_key in self.cache:
            logger.debug(f"Cache hit for query (key: {cache_key})")
            return self.cache[cache_key]
        
        # Perform reranking
        reranked = rerank_documents(documents, query, top_k)
        
        # Store in cache (with size limit)
        if len(self.cache) >= self.max_size:
            self.cache.pop(next(iter(self.cache)))  # Remove oldest
        
        self.cache[cache_key] = reranked
        logger.debug(f"Cached reranking result (key: {cache_key})")
        
        return reranked

# Global cache instance
_reranker_cache = RerankerCache(max_cache_size=100)

def rerank_documents_cached(
    documents: List[Document],
    query: str,
    top_k: int = TOP_K_RERANKED
) -> List[Document]:
    """Use cached reranker."""
    return _reranker_cache.get_reranked(documents, query, top_k)
```

### Strategy 3: Async Reranking

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

async def rerank_documents_async(
    documents: List[Document],
    query: str,
    top_k: int = TOP_K_RERANKED
) -> List[Document]:
    """Asynchronous document reranking."""
    loop = asyncio.get_event_loop()
    
    def _rerank():
        return rerank_documents(documents, query, top_k)
    
    result = await loop.run_in_executor(executor, _rerank)
    return result

# Use in async endpoint:
from fastapi import FastAPI
from fastapi.responses import JSONResponse

@app.get("/ask-async")
async def ask_async(
    question: str = Query(...),
    session_id: str = Query(default="default")
):
    """Async endpoint with reranking."""
    try:
        rag_chain = get_rag_chain()
        answer = await asyncio.to_thread(
            rag_chain.invoke,
            {"question": question},
            {"configurable": {"session_id": session_id}}
        )
        return JSONResponse({"answer": answer})
    except Exception as exc:
        logger.exception("Error in async ask endpoint")
        raise HTTPException(status_code=500, detail=str(exc))
```

---

## Advanced Pipeline Configurations

### Configuration 1: Multi-Stage Reranking

First stage: Fast lightweight reranker
Second stage: Heavyweight reranker for top candidates

```python
def rerank_documents_multistage(
    documents: List[Document],
    query: str,
    top_k: int = TOP_K_RERANKED
) -> List[Document]:
    """
    Two-stage reranking:
    1. Fast stage: lightweight model filters to top-50%
    2. Slow stage: high-quality model selects final top-k
    """
    if len(documents) <= top_k:
        return documents
    
    from sentence_transformers import CrossEncoder
    
    # Stage 1: Fast filter
    logger.debug("Stage 1: Fast filtering...")
    fast_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
    pairs = [[query, doc.page_content] for doc in documents]
    fast_scores = fast_reranker.predict(pairs)
    
    # Keep top 50%
    intermediate_k = max(top_k * 2, int(len(documents) * 0.5))
    intermediate_docs = sorted(
        zip(documents, fast_scores),
        key=lambda x: x[1],
        reverse=True
    )[:intermediate_k]
    
    logger.debug(f"Stage 1: Filtered {len(documents)} → {len(intermediate_docs)} documents")
    
    # Stage 2: High-quality reranker
    logger.debug("Stage 2: High-quality reranking...")
    high_quality_reranker = CrossEncoder("BAAI/bge-reranker-large")
    intermediate_docs_only = [doc for doc, _ in intermediate_docs]
    pairs_stage2 = [[query, doc.page_content] for doc in intermediate_docs_only]
    high_scores = high_quality_reranker.predict(pairs_stage2)
    
    final_docs = sorted(
        zip(intermediate_docs_only, high_scores),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]
    
    logger.debug(f"Stage 2: Selected {len(final_docs)} final documents")
    
    return [doc for doc, _ in final_docs]
```

### Configuration 2: Context-Aware Reranking

Rerank differently based on document type or category:

```python
def rerank_documents_contextaware(
    documents: List[Document],
    query: str,
    doc_type: str = "general",  # "technical", "general", "legal", etc.
    top_k: int = TOP_K_RERANKED
) -> List[Document]:
    """
    Use different reranker models based on document type.
    """
    model_mapping = {
        "technical": "cross-encoder/qnli-distilroberta-base",
        "legal": "BAAI/bge-reranker-large",
        "general": "BAAI/bge-reranker-base",
    }
    
    model_name = model_mapping.get(doc_type, "BAAI/bge-reranker-base")
    reranker = CrossEncoder(model_name)
    
    logger.info(f"Using {model_name} for {doc_type} documents")
    
    pairs = [[query, doc.page_content] for doc in documents]
    scores = reranker.predict(pairs)
    
    ranked = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]
    
    return [doc for doc, _ in ranked]
```

---

## Monitoring & Analytics

### Detailed Reranking Metrics

```python
from dataclasses import dataclass
from typing import List
import time

@dataclass
class RerangingMetrics:
    query: str
    initial_count: int
    final_count: int
    latency_ms: float
    avg_score: float
    score_range: tuple  # (min, max)
    top_scores: List[float]

def rerank_documents_with_metrics(
    documents: List[Document],
    query: str,
    top_k: int = TOP_K_RERANKED
) -> tuple[List[Document], RerangingMetrics]:
    """Rerank documents and return detailed metrics."""
    start_time = time.time()
    
    reranker = get_reranker()
    pairs = [[query, doc.page_content] for doc in documents]
    scores = reranker.predict(pairs)
    
    ranked_docs = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]
    
    latency_ms = (time.time() - start_time) * 1000
    
    # Calculate metrics
    final_scores = [score for _, score in ranked_docs]
    metrics = RerangingMetrics(
        query=query,
        initial_count=len(documents),
        final_count=len(ranked_docs),
        latency_ms=latency_ms,
        avg_score=sum(scores) / len(scores) if scores else 0,
        score_range=(min(scores), max(scores)) if scores else (0, 0),
        top_scores=final_scores
    )
    
    logger.info(f"Reranking metrics: {metrics}")
    
    return [doc for doc, _ in ranked_docs], metrics

# Usage with tracking
def get_rag_chain_with_metrics():
    """Enhanced RAG chain that tracks reranking metrics."""
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    
    def format_docs(documents: List[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in documents)
    
    def retrieve_and_rerank_with_metrics(state: dict):
        question = state["question"]
        retrieved_docs = retriever.invoke(question)
        reranked_docs, metrics = rerank_documents_with_metrics(
            retrieved_docs, question, top_k=TOP_K_RERANKED
        )
        # Store metrics in state for logging
        state["reranking_metrics"] = metrics
        return reranked_docs
    
    # ... rest of chain setup
```

### Analytics Dashboard Data

```python
from datetime import datetime, timedelta
from collections import defaultdict

class RerangingAnalytics:
    def __init__(self):
        self.query_history = []
        self.latency_history = []
        self.score_history = []
    
    def record_reranking(self, metrics: RerangingMetrics):
        """Record reranking event."""
        self.query_history.append({
            "timestamp": datetime.now(),
            "query": metrics.query,
            "initial_count": metrics.initial_count,
            "final_count": metrics.final_count
        })
        self.latency_history.append(metrics.latency_ms)
        self.score_history.append(metrics.avg_score)
    
    def get_stats(self, window_minutes: int = 60) -> dict:
        """Get aggregated statistics."""
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        recent_queries = [q for q in self.query_history if q["timestamp"] > cutoff]
        recent_latencies = self.latency_history[-len(recent_queries):]
        recent_scores = self.score_history[-len(recent_queries):]
        
        return {
            "total_queries": len(recent_queries),
            "avg_latency_ms": sum(recent_latencies) / len(recent_latencies) if recent_latencies else 0,
            "p95_latency_ms": sorted(recent_latencies)[int(len(recent_latencies) * 0.95)] if recent_latencies else 0,
            "avg_score": sum(recent_scores) / len(recent_scores) if recent_scores else 0,
            "window_minutes": window_minutes
        }

_analytics = RerangingAnalytics()

@app.get("/reranking-stats")
def get_reranking_stats():
    """Endpoint to view reranking performance stats."""
    return _analytics.get_stats()
```

---

## Production Deployment

### Environment Configuration

```python
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration from environment
RERANKER_MODEL_NAME = os.getenv(
    "RERANKER_MODEL_NAME",
    "BAAI/bge-reranker-base"
)
TOP_K_RERANKED = int(os.getenv("TOP_K_RERANKED", "3"))
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "cuda")  # "cuda" or "cpu"
RERANKER_BATCH_SIZE = int(os.getenv("RERANKER_BATCH_SIZE", "32"))

def get_reranker_production() -> CrossEncoder:
    """Production-grade reranker initialization with device selection."""
    global _RERANKER
    
    if _RERANKER is None:
        logger.info(
            f"Loading reranker: {RERANKER_MODEL_NAME} "
            f"on device: {RERANKER_DEVICE}"
        )
        _RERANKER = CrossEncoder(
            RERANKER_MODEL_NAME,
            device=RERANKER_DEVICE,
            automodel_args={"torch_dtype": "float16"}  # Reduce memory
        )
    
    return _RERANKER
```

### `.env` Example

```env
# Production Environment Variables
RERANKER_MODEL_NAME=BAAI/bge-reranker-base
TOP_K_RERANKED=3
RERANKER_DEVICE=cuda
RERANKER_BATCH_SIZE=32
OLLAMA_BASE_URL=http://localhost:11434
LOG_LEVEL=INFO
```

### Docker Integration

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-download reranker model
RUN python -c "from sentence_transformers import CrossEncoder; \
    CrossEncoder('BAAI/bge-reranker-base')"

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Health Check with Reranker Verification

```python
from fastapi import HTTPException

@app.get("/health")
async def health_check():
    """Health check that verifies reranker is loaded."""
    try:
        reranker = get_reranker()
        
        # Quick test
        test_docs = [["test query", "test document"]]
        _ = reranker.predict(test_docs)
        
        return {
            "status": "healthy",
            "reranker": "ready",
            "model": RERANKER_MODEL_NAME
        }
    except Exception as exc:
        logger.error(f"Health check failed: {exc}")
        raise HTTPException(status_code=503, detail="Reranker not ready")
```

---

## Summary of Customization Options

| Feature | Implementation | Use Case |
|---------|---|---|
| Weighted Scoring | `rerank_documents_weighted()` | Balance relevance with metadata |
| Threshold Filtering | `rerank_documents_threshold()` | Quality-first approach |
| Diversity | `rerank_documents_diverse()` | Avoid redundant results |
| Batch Processing | `rerank_documents_batch()` | High-throughput scenarios |
| Caching | `RerankerCache` | Reduce repeated computation |
| Async | `rerank_documents_async()` | Non-blocking requests |
| Multi-Stage | `rerank_documents_multistage()` | Optimize latency-quality tradeoff |
| Context-Aware | `rerank_documents_contextaware()` | Domain-specific reranking |
| Analytics | `RerangingAnalytics` | Performance monitoring |

Choose based on your specific requirements!

