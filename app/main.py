from functools import lru_cache

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from app.ingestion.pdf_loader import extract_pdf_pages
from app.retrieval.chunker import chunk_pages
from app.retrieval.embedder import Embedder
from app.retrieval.vector_store import VectorStore


app = FastAPI(
     title="Multimodal AI Enterprise Copilot",
    description="An AI assistant for documents and structured data.",
    version="0.3.0",
)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
vector_store = VectorStore()

@lru_cache(maxsize=1)
def get_embedder():
    # Load once and reuse the model.
    return Embedder()


class SearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)


@app.get("/")
def home():
    return {
        "message": "Multimodal AI Copilot is running!",
        "status": "success",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    filename = file.filename or "document.pdf"

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    try:
        content = await file.read(MAX_FILE_SIZE + 1)

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="PDF exceeds the 10 MB size limit.",
            )

        pages = extract_pdf_pages(content, filename)
        chunks = chunk_pages(pages)
        
        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="No extractable text found. The PDF may require OCR.",
            )
            
        embeddings = get_embedder().encode_documents(
            [chunk["text"] for chunk in chunks]
        )

        vector_store.add(chunks, embeddings)

        return {
            "filename": filename,
            "total_pages": len(pages),
            "pages_with_text": sum(
                page["has_text"] for page in pages
            ),
            "chunks_created": len(chunks),
            "embedding_dimensions": int(embeddings.shape[1]),
            "message": "Document indexed successfully.",
        }


    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    finally:
        await file.close()


@app.post("/search")
def search_documents(request: SearchRequest):
    if not request.question.strip():
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty.",
        )

    if not vector_store.chunks:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed. Upload a PDF first.",
        )

    query_embedding = get_embedder().encode_query(
        request.question
    )

    results = vector_store.search(
        query_embedding,
        top_k=request.top_k,
    )

    return {
        "question": request.question,
        "results": results,
    }
