"""CV Agent graph nodes. Each node: takes state → returns state updates."""

import os
import json

from dotenv import load_dotenv
from openai import OpenAI

from app.cv_agent.retriever import multi_search

load_dotenv()

CHAT_MODEL = "gpt-4o"


def _call_gpt(system: str, user: str) -> str:
    """Simple GPT call — used by all nodes."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content


# ═══════════════════════════════════════════════════════════════════════
# Node 1: ROUTER — breaks user request into search queries
# ═══════════════════════════════════════════════════════════════════════

ROUTER_SYSTEM = """\
You break down a user's CV-related request into specific search queries.

The user has interview transcripts stored as chunks. You need to generate
search queries that will find the most relevant chunks.

Rules:
- Return a JSON array of 3-5 search query strings
- Each query MUST be specific to the topic requested — not generic
- Queries should be in Hebrew (the language of the transcripts)
- Think about what the user likely SAID in casual conversation, not formal terms
- NEVER use generic queries like "פתרון בעיות" or "עבודה בלחץ" or "ניהול צוותים" — these match everything and are useless
- Each query should mention concrete things: project names, technologies, job titles, specific activities
- Return ONLY the JSON array, nothing else

Example:
Request: "Write a CV section about my Full Stack experience"
Output: ["פיתוח Full Stack ו-React", "עבודה עם Docker ו-AWS ו-S3", "APIs ואינטגרציות טלגרם", "תיקון באגים במערכת קיימת"]

Example:
Request: "Write a CV section about working in the exam department at Bezalel"
Output: ["עבודה במחלקת מבחנים בצלאל", "ניהול מבחנים וקבלה בבצלאל", "עבודה עם מידע רגיש של סטודנטים", "תיאום מבחנים עם מרצים"]
"""


def router_node(state: dict) -> dict:
    """Break user request into multiple search queries."""
    result = _call_gpt(ROUTER_SYSTEM, state["user_request"])

    # Parse JSON array from response
    try:
        queries = json.loads(result)
    except json.JSONDecodeError:
        # Fallback: use original request as single query
        queries = [state["user_request"]]

    return {"search_queries": queries}


# ═══════════════════════════════════════════════════════════════════════
# Node 2: RETRIEVE — fetch chunks using search queries
# ═══════════════════════════════════════════════════════════════════════

def retrieve_node(state: dict) -> dict:
    """Run all search queries and merge results."""
    queries = state["search_queries"]
    min_score = state.get("min_score", 0.38)

    chunks = multi_search(queries, top_k=10, min_score=min_score)

    return {
        "chunks": chunks,
        "iteration": state.get("iteration", 0) + 1,
    }


# ═══════════════════════════════════════════════════════════════════════
# Node 3: EVALUATE — check if we have enough data
# ═══════════════════════════════════════════════════════════════════════

EVALUATE_SYSTEM = """\
You evaluate whether retrieved transcript chunks contain enough information
to fulfill a CV-related request.

Rules:
- Answer with a JSON object: {"sufficient": true/false, "missing": "...", "new_queries": [...]}
- "sufficient": true if the chunks contain concrete, specific information for the request
- "missing": describe what's missing (empty string if sufficient)
- "new_queries": if not sufficient, suggest 2-3 new search queries in Hebrew to find what's missing
- Return ONLY the JSON object
"""


def evaluate_node(state: dict) -> dict:
    """Check if retrieved chunks are sufficient for the request."""
    chunks_summary = "\n\n".join(
        f"[{c['chunk_id']} | score={c['score']:.3f}]\n{c['text'][:500]}"
        for c in state["chunks"]
    )

    prompt = f"""Request: {state['user_request']}

Retrieved {len(state['chunks'])} chunks:

{chunks_summary}

Are these chunks sufficient to write a good response?"""

    result = _call_gpt(EVALUATE_SYSTEM, prompt)

    try:
        evaluation = json.loads(result)
    except json.JSONDecodeError:
        evaluation = {"sufficient": True, "missing": "", "new_queries": []}

    if not evaluation.get("sufficient") and state.get("iteration", 1) < state.get("max_iterations", 3):
        # Add new queries and signal more data needed
        new_queries = evaluation.get("new_queries", [])
        existing_queries = state.get("search_queries", [])
        return {
            "search_queries": existing_queries + new_queries,
            "needs_more_data": True,
        }

    return {"needs_more_data": False}


# ═══════════════════════════════════════════════════════════════════════
# Node 4: GENERATE — write CV content from chunks
# ═══════════════════════════════════════════════════════════════════════

GENERATE_SYSTEM = """\
You are a career assistant that writes CV content based on interview transcripts.

STRICT RULES:
- Use ONLY information found in the provided transcript chunks
- If a fact is not in the chunks, DO NOT include it
- If there is not enough information for a section, say so explicitly
- Write in the language the user requested
- Be concise and professional
- Format for a CV — bullet points, action verbs, measurable results where available
- Do not add generic filler phrases
- When specific technologies are mentioned in the chunks (e.g. S3, EC2, React, Next.js, FastAPI, Docker, PostgreSQL, MongoDB, LangChain, LangGraph, Telethon), include them by name — do not generalize to "cloud services" or "various technologies"
- List technologies in a dedicated "Technologies" section when relevant
"""


def generate_node(state: dict) -> dict:
    """Generate CV content from retrieved chunks."""
    chunks_text = "\n\n---\n\n".join(
        f"[{c['chunk_id']}]\n{c['text']}" for c in state["chunks"]
    )

    prompt = f"""## Source material (interview transcript chunks):

{chunks_text}

## Request:

{state['user_request']}"""

    draft = _call_gpt(GENERATE_SYSTEM, prompt)
    return {"draft": draft}


# ═══════════════════════════════════════════════════════════════════════
# Node 5: VERIFY — fact-check draft against chunks
# ═══════════════════════════════════════════════════════════════════════

VERIFY_SYSTEM = """\
You are a fact-checker. You compare a CV draft against source transcript chunks.

Your job:
1. Check every claim in the draft against the chunks
2. Remove or flag anything not supported by the chunks
3. Return the corrected version

Rules:
- If a claim has no supporting chunk, REMOVE it
- If a claim is exaggerated, tone it down to match what was actually said
- Keep the same format and language as the draft
- Add a "---" separator at the end, followed by a brief note listing what you removed/changed (if anything)
- If everything checks out, return the draft as-is with a note: "No changes needed"
"""


def verify_node(state: dict) -> dict:
    """Cross-check draft against source chunks. Remove unsupported claims."""
    chunks_text = "\n\n---\n\n".join(
        f"[{c['chunk_id']}]\n{c['text']}" for c in state["chunks"]
    )

    prompt = f"""## Draft to verify:

{state['draft']}

## Source chunks:

{chunks_text}"""

    verified = _call_gpt(VERIFY_SYSTEM, prompt)
    return {"verified_output": verified, "done": True}
