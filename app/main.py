from fastapi import FastAPI, File, HTTPException, UploadFile
from app.ingestion.pdf_loader import extract_pdf_pages

app = FastAPI(
     title="Multimodal AI Enterprise Copilot",
    description="An AI assistant for documents and structured data.",
    version="0.1.0",
)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

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

        extracted_text_pages = sum(
            page["has_text"] for page in pages
        )

        return {
            "filename": filename,
            "total_pages": len(pages),
            "pages_with_text": extracted_text_pages,
            "pages_needing_ocr": len(pages) - extracted_text_pages,
            "pages": pages,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    finally:
        await file.close()
