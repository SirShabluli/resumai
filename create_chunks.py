"""
Create conversation chunks from messages and generate embeddings.
Usage: python create_chunks.py
"""

import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from openai import OpenAI
from pymongo import MongoClient

load_dotenv()

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "career_app"
EMBEDDING_MODEL = "text-embedding-3-small"

# Chunking params
WINDOW_SIZE = 6      # turns per chunk (smaller = more focused embeddings)
OVERLAP = 2          # turns shared between consecutive chunks
MAX_CHARS = 6000     # safe limit to stay under 8192 tokens


def get_messages():
    """Read all messages from MongoDB, sorted by message_index."""
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    messages = list(db.messages.find().sort("message_index", 1))
    client.close()
    return messages


def group_by_session(messages):
    """Group messages by session_key so chunks don't cross sessions."""
    sessions = {}
    for msg in messages:
        key = msg.get("session_key", "unknown")
        sessions.setdefault(key, []).append(msg)
    return sessions


def build_chunk(window, session_key):
    """Build a single chunk dict from a list of messages."""
    lines = [f"{msg['role']}: {msg['text']}" for msg in window]
    return {
        "logical_conversation_id": window[0].get("logical_conversation_id"),
        "session_key": session_key,
        "message_ids": [msg["_id"] for msg in window],
        "start_message_index": window[0]["message_index"],
        "end_message_index": window[-1]["message_index"],
        "text": "\n".join(lines),
        "turn_count": len(window),
        "source_file": window[0].get("source_file"),
    }


def make_chunks(messages_in_session, session_key):
    """Sliding window over messages within one session.
    If a chunk exceeds MAX_CHARS, shrink window until it fits."""
    chunks = []
    i = 0
    while i < len(messages_in_session):
        # Start with full window, shrink if text too long
        end = min(i + WINDOW_SIZE, len(messages_in_session))
        chunk = build_chunk(messages_in_session[i:end], session_key)

        while len(chunk["text"]) > MAX_CHARS and end > i + 1:
            end -= 1
            chunk = build_chunk(messages_in_session[i:end], session_key)

        chunks.append(chunk)

        # Advance by actual window size minus overlap
        step = max((end - i) - OVERLAP, 1)
        i += step

    return chunks


def create_embeddings(chunks, openai_client):
    """Call OpenAI embeddings API in batches of 50."""
    BATCH_SIZE = 50
    print(f"Requesting embeddings for {len(chunks)} chunks in batches of {BATCH_SIZE}...")

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        texts = [c["text"] for c in batch]

        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )

        for i, item in enumerate(response.data):
            chunks[start + i]["embedding"] = item.embedding
            chunks[start + i]["embedding_model"] = EMBEDDING_MODEL

        print(f"  Batch {start // BATCH_SIZE + 1}: {len(batch)} chunks done")

    return chunks


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in .env")
        return

    # 1. Read messages
    messages = get_messages()
    print(f"Loaded {len(messages)} messages from MongoDB")

    # 2. Group by session
    sessions = group_by_session(messages)
    print(f"Found {len(sessions)} sessions: {list(sessions.keys())}")

    # 3. Create chunks
    all_chunks = []
    for session_key, session_msgs in sessions.items():
        session_chunks = make_chunks(session_msgs, session_key)
        all_chunks.extend(session_chunks)

    print(f"Created {len(all_chunks)} chunks (window={WINDOW_SIZE}, overlap={OVERLAP})")

    # 4. Generate embeddings
    openai_client = OpenAI(api_key=api_key)
    all_chunks = create_embeddings(all_chunks, openai_client)

    # 5. Add chunk_id and timestamp
    for i, chunk in enumerate(all_chunks):
        chunk["chunk_id"] = f"{chunk['logical_conversation_id']}_chunk_{i+1:03d}"
        chunk["created_at"] = datetime.now(timezone.utc)

    # 6. Save to MongoDB
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]

    existing = db.chunks.count_documents({})
    if existing > 0:
        print(f"Dropping existing {existing} chunks and re-creating.")
        db.chunks.drop()

    db.chunks.insert_many(all_chunks)
    print(f"Saved {len(all_chunks)} chunks to {DB_NAME}.chunks")

    # 7. Quick sanity check
    sample = db.chunks.find_one({}, {"chunk_id": 1, "turn_count": 1, "text": 1})
    print(f"\nSample chunk: {sample['chunk_id']} ({sample['turn_count']} turns)")
    print(f"Text preview: {sample['text'][:150]}...")

    client.close()


if __name__ == "__main__":
    main()
