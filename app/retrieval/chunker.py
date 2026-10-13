
def chunk_pages(
    pages: list[dict],
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[dict]:
    """
    Split extracted PDF page text into smaller overlapping chunks.

    Each chunk retains its source filename and page number.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    chunks = []
    step = chunk_size - overlap

    for page in pages:
        text = page.get("text", "").strip()

        # Skip empty pages, such as scanned pages without OCR.
        if not text:
            continue

        for start in range(0, len(text), step):
            chunk_text = text[start : start + chunk_size].strip()

            if not chunk_text:
                continue

            chunks.append(
                {
                    "source": page["source"],
                    "page": page["page"],
                    "chunk_index": len(chunks),
                    "text": chunk_text,
                }
            )

            # Stop when the end of this page's text is reached.
            if start + chunk_size >= len(text):
                break

    return chunks
