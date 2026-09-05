"""Retrieve relevant chunks from MongoDB using vector similarity."""

import os

from dotenv import load_dotenv
from openai import OpenAI
from pymongo import MongoClient

load_dotenv()

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "career_app"
EMBEDDING_MODEL = "text-embedding-3-small"


def _cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _load_all_chunks():
    """Load all chunks with embeddings from MongoDB."""
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    chunks = list(db.chunks.find({}, {
        "chunk_id": 1, "text": 1, "embedding": 1,
        "session_key": 1, "turn_count": 1,
        "start_message_index": 1, "end_message_index": 1,
    }))
    client.close()
    return chunks


def search_chunks(query: str, top_k: int = 10, min_score: float = 0.30) -> list[dict]:
    """Search chunks by semantic similarity to a query string.

    Returns list of {chunk_id, text, score, session_key, ...} — no embedding.
    """
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Embed the query
    response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=query)
    query_vec = response.data[0].embedding

    # Score all chunks
    all_chunks = _load_all_chunks()
    scored = []
    for chunk in all_chunks:
        score = _cosine_similarity(query_vec, chunk["embedding"])
        if score >= min_score:
            scored.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "score": score,
                "session_key": chunk.get("session_key"),
                "start_message_index": chunk.get("start_message_index"),
                "end_message_index": chunk.get("end_message_index"),
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def multi_search(queries: list[str], top_k: int = 10, min_score: float = 0.30) -> list[dict]:
    """Run multiple queries and merge results, removing duplicates.

    Each query contributes its top results. Duplicates keep the highest score.
    """
    seen = {}  # chunk_id -> result dict

    for query in queries:
        results = search_chunks(query, top_k=top_k, min_score=min_score)
        for r in results:
            cid = r["chunk_id"]
            if cid not in seen or r["score"] > seen[cid]["score"]:
                seen[cid] = r

    merged = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
    return merged
