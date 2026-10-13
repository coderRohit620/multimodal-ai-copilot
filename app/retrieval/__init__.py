
def chunk_pages(
    pages: list[dict],
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[dict]:
    """
    Split extracted page text into overlapping chunks.

    Each chunk preserves its source filename and page number.
    Chunk sizes are measured in characters.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    chunks = []
    step = chunk_size - overlap

    for page in pages:
        text = page.get("text", "").strip()

        if not text:
            continue

        for start in range(0, len(text), step):
            chunk_text = text[start:start + chunk_size].strip()

            if not chunk_text:
                continue

            chunks.append({
                "source": page["source"],
                "page": page["page"],
                "chunk_index": len(chunks),
                "text": chunk_text,
            })

            if start + chunk_size >= len(text):
                break

    return chunks
