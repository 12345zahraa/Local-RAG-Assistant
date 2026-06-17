from __future__ import annotations

import logging
import os
import sys
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel
from pypdf import PdfReader

# ترفند طلایی: اضافه کردن خودکار مسیر پروژه به پایتون
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from app.core_rag import get_rag_chain, process_and_index_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Local RAG API",
    description="A production-style local retrieval and generation API powered by FastAPI, Ollama, and Qdrant.",
    version="1.0.0",
)


class IndexResponse(BaseModel):
    message: str
    chunk_count: int


class AskResponse(BaseModel):
    answer: str


@app.post(
    "/index-document",
    response_model=IndexResponse,
    summary="Upload a PDF file and index its text into the local vector store",
)
async def index_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    try:
        contents = await file.read()
        reader = PdfReader(BytesIO(contents))
        text_parts = []
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text_parts.append(extracted)

        text = "\n\n".join(text_parts).strip()
        if not text:
            raise ValueError("No readable text found in the uploaded PDF.")

        chunk_count = process_and_index_text(text)
        return IndexResponse(
            message="Document indexed successfully.",
            chunk_count=chunk_count,
        )
    except ValueError as exc:
        logger.warning("PDF indexing validation failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("PDF indexing runtime error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while indexing PDF")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error while indexing document: {exc}",
        ) from exc


@app.get(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question using the local RAG pipeline",
)
def ask(question: str = Query(..., min_length=1, description="Question to ask the model.")):
    try:
        rag_chain = get_rag_chain()
        answer = rag_chain.invoke(question)
        return AskResponse(answer=answer)
    except RuntimeError as exc:
        logger.warning("Ask request failed: %s", exc)
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while generating answer")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error while generating answer: {exc}",
        ) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)