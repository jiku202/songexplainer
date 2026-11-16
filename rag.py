"""Retrieval over the curated knowledge base (document loader + TF-IDF vector index).

Each file in knowledge_base/ is a small markdown document with a front-matter header:

    ---
    title: Fix You
    artist: Coldplay
    year: 2005
    tags: grief, comfort, ballad
    ---
    Body text in paragraphs...

Documents are chunked by paragraph, each chunk is prefixed with its metadata, and the
chunks are embedded as TF-IDF vectors (word 1-2 grams). Retrieval is cosine similarity.
This keeps the app dependency-light and free of model downloads, so it deploys cleanly
on Streamlit Community Cloud. To switch to neural embeddings, replace SongKnowledgeBase
with a ChromaDB collection exposing the same .search() method.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB_DIR = Path(__file__).parent / "knowledge_base"


@dataclass
class Chunk:
    title: str
    artist: str
    year: str
    tags: str
    text: str
    source: str

    @property
    def index_text(self) -> str:
        return f"{self.title} {self.artist} {self.year} {self.tags} {self.text}"


def parse_document(path: Path) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8").strip()
    meta: dict[str, str] = {}
    body = raw
    if raw.startswith("---"):
        _, header, body = raw.split("---", 2)
        for line in header.strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip().lower()] = v.strip()
    paragraphs = [p.strip() for p in body.strip().split("\n\n") if p.strip()]
    return [
        Chunk(meta.get("title", path.stem), meta.get("artist", ""), meta.get("year", ""),
              meta.get("tags", ""), p, path.name)
        for p in paragraphs
    ]


def load_chunks(kb_dir: Path = KB_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.glob("*.md")):
        chunks.extend(parse_document(path))
    return chunks


class SongKnowledgeBase:
    def __init__(self, kb_dir: Path = KB_DIR):
        self.chunks = load_chunks(kb_dir)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.matrix = self.vectorizer.fit_transform([c.index_text for c in self.chunks]) if self.chunks else None

    def search(self, query: str, k: int = 3, min_score: float = 0.08) -> list[tuple[Chunk, float]]:
        if not self.chunks or not query.strip():
            return []
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        ranked = sorted(zip(self.chunks, scores), key=lambda x: x[1], reverse=True)[:k]
        return [(c, float(s)) for c, s in ranked if s >= min_score]

    def format_results(self, query: str, k: int = 3) -> str:
        hits = self.search(query, k)
        if not hits:
            return "No relevant notes found in the knowledge base."
        return "\n\n".join(
            f"[{c.title} - {c.artist}, {c.year} | source: {c.source} | score {s:.2f}]\n{c.text}"
            for c, s in hits
        )
