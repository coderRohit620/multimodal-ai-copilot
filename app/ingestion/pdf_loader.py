
from pathlib import Path

import pymupdf


def extract_pdf_pages(
    pdf_bytes: bytes,
    filename: str,
) -> list[dict]:
    """
    Extract text from each page of a PDF.

    Page numbers are one-based so they match the
    page numbers users see in a PDF reader.
    """
    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")

    if not pdf_bytes.startswith(b"%PDF-"):
        raise ValueError("The uploaded file is not a valid PDF.")

    safe_filename = Path(filename).name

    try:
        with pymupdf.open(
            stream=pdf_bytes,
            filetype="pdf",
        ) as pdf:
            if pdf.page_count == 0:
                raise ValueError("The PDF contains no pages.")

            pages = []

            for page_index, page in enumerate(pdf):
                text = page.get_text("text").strip()

                pages.append(
                    {
                        "source": safe_filename,
                        "page": page_index + 1,
                        "text": text,
                        "has_text": bool(text),
                    }
                )

            return pages

    except pymupdf.FileDataError as exc:
        raise ValueError(
            "The PDF could not be opened. "
            "It may be damaged or invalid."
        ) from exc
