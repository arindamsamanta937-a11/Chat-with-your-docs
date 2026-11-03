import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")  # free, runs locally


def load_pdf(file) -> str:
    reader = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(text, size=800, overlap=150):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def build_index(chunks):
    return embedder.encode(chunks, normalize_embeddings=True)


def retrieve(query, chunks, index, k=4):
    q = embedder.encode([query], normalize_embeddings=True)[0]
    scores = index @ q
    top = np.argsort(scores)[::-1][:k]
    return [chunks[i] for i in top]