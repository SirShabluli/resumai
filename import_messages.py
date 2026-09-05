"""
Import career interview messages from JSONL into MongoDB.
Usage: python import_messages.py <path_to_jsonl>
"""

import json
import sys
from pymongo import MongoClient

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "career_app"
COLLECTION_NAME = "messages"


def main():
    if len(sys.argv) < 2:
        print("Usage: python import_messages.py <path_to_jsonl>")
        sys.exit(1)

    jsonl_path = sys.argv[1]

    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    # Avoid duplicates on re-run: clear existing messages
    existing = collection.count_documents({})
    if existing > 0:
        print(f"Collection already has {existing} documents. Dropping and re-importing.")
        collection.drop()

    docs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))

    collection.insert_many(docs)
    print(f"Imported {len(docs)} messages into {DB_NAME}.{COLLECTION_NAME}")

    client.close()


if __name__ == "__main__":
    main()
