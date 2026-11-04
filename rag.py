import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")  # free, runs locally


def load_pdf(file, filename):
    """Return a list of pages: {"text", "page", "source"}."""
    reader = PdfReader(file)
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"text": text, "page": number, "source": filename})
    return pages


def chunk_pages(pages, size=800, overlap=150):
    """Chunk each page separately so every chunk keeps its page number."""
    chunks = []
    for p in pages:
        text = p["text"]
        start = 0
        while start < len(text):
            piece = text[start:start + size].strip()
            if piece:
                chunks.append({"text": piece, "page": p["page"], "source": p["source"]})
            start += size - overlap
    return chunks


def build_index(chunks):
    return embedder.encode([c["text"] for c in chunks], normalize_embeddings=True)


def retrieve(query, chunks, index, k=4, min_score=0.25):
    q = embedder.encode([query], normalize_embeddings=True)[0]
    scores = index @ q
    top = np.argsort(scores)[::-1][:k]
    return [chunks[i] for i in top if scores[i] >= min_score]