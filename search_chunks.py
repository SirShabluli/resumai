"""
Search chunks by semantic similarity.
Usage: python search_chunks.py "ניסיון עם Docker"
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI
from pymongo import MongoClient

load_dotenv()

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "career_app"
EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K = 5


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def main():
    if len(sys.argv) < 2:
        print('Usage: python search_chunks.py "your query here"')
        sys.exit(1)

    query = sys.argv[1]
    print(f"Query: {query}\n")

    # 1. Embed the query
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_vec = response.data[0].embedding

    # 2. Load all chunks from MongoDB
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    chunks = list(db.chunks.find({}, {"chunk_id": 1, "text": 1, "embedding": 1, "session_key": 1}))
    client.close()

    # 3. Score each chunk
    scored = []
    for chunk in chunks:
        score = cosine_similarity(query_vec, chunk["embedding"])
        scored.append((score, chunk))

    # 4. Sort and show top results
    scored.sort(key=lambda x: x[0], reverse=True)

    print(f"Top {TOP_K} results:\n" + "=" * 60)
    for rank, (score, chunk) in enumerate(scored[:TOP_K], 1):
        print(f"\n#{rank}  score={score:.4f}  [{chunk['chunk_id']}]  session={chunk['session_key']}")
        print("-" * 60)
        # Show first 300 chars of text
        preview = chunk["text"][:300]
        print(preview)
        if len(chunk["text"]) > 300:
            print("...")
        print()


if __name__ == "__main__":
    main()
