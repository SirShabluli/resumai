"""
Streamlit UI for searching and generating from conversation chunks.
Usage: streamlit run search_ui.py
"""

import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pymongo import MongoClient

from app.cv_agent.graph import cv_agent

load_dotenv()

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "career_app"
EMBEDDING_MODEL = "text-embedding-3-small"


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@st.cache_resource
def get_clients():
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    mongo_client = MongoClient(MONGO_URI)
    return openai_client, mongo_client


@st.cache_data(ttl=300)
def load_chunks():
    _, mongo_client = get_clients()
    db = mongo_client[DB_NAME]
    return list(db.chunks.find({}, {"chunk_id": 1, "text": 1, "embedding": 1, "session_key": 1, "turn_count": 1, "start_message_index": 1, "end_message_index": 1}))


def search(query_vec, top_k):
    chunks = load_chunks()
    scored = [(cosine_similarity(query_vec, c["embedding"]), c) for c in chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def format_chunk_text(text):
    return text.replace("\nuser:", "\n\nuser:").replace("\nassistant:", "\n\nassistant:")


# --- UI ---

st.set_page_config(page_title="ResumAI", layout="wide")
st.title("ResumAI")

tab_search, tab_generate = st.tabs(["Search", "Generate (Agent)"])

# --- Search Tab ---
with tab_search:
    query = st.text_input("Search query", placeholder="e.g. ניסיון עם Docker ו-AWS", key="search_q")
    top_k = st.slider("Results", min_value=1, max_value=20, value=5)

    if query:
        openai_client, _ = get_clients()
        response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=query)
        query_vec = response.data[0].embedding

        results = search(query_vec, top_k)

        st.markdown(f"**{len(load_chunks())} chunks searched**")
        st.divider()

        for rank, (score, chunk) in enumerate(results, 1):
            with st.expander(f"#{rank}  |  score: {score:.4f}  |  {chunk['chunk_id']}  |  session: {chunk['session_key']}", expanded=rank <= 3):
                st.caption(f"Messages {chunk['start_message_index']}–{chunk['end_message_index']}  |  {chunk['turn_count']} turns")
                st.markdown(format_chunk_text(chunk["text"]))

# --- Generate Tab (Agent) ---
with tab_generate:
    prompt = st.text_area("What do you want to generate?", placeholder="e.g. תכתוב סעיף CV על הדרכת טיולים בספרד", key="gen_q", height=100)

    if st.button("Generate", type="primary") and prompt:
        with st.spinner("Agent working: routing → retrieving → evaluating → generating → verifying..."):
            result = cv_agent.invoke({
                "user_request": prompt,
                "search_queries": [],
                "chunks": [],
                "min_score": 0.35,
                "draft": "",
                "verified_output": "",
                "iteration": 0,
                "max_iterations": 3,
                "needs_more_data": False,
                "done": False,
            })

        # Show verified output
        st.markdown("### Result (verified)")
        st.markdown(result["verified_output"])

        # Show draft before verification
        with st.expander("Draft (before verification)"):
            st.markdown(result["draft"])

        # Show search queries used
        with st.expander(f"Search queries ({len(result['search_queries'])})"):
            for q in result["search_queries"]:
                st.markdown(f"- {q}")

        # Show chunks used
        with st.expander(f"Chunks used ({len(result['chunks'])})"):
            for c in result["chunks"]:
                st.caption(f"{c['chunk_id']} | score={c['score']:.3f} | session={c['session_key']}")
                st.markdown(format_chunk_text(c["text"][:300] + "..."))
                st.divider()

        # Show iterations
        st.caption(f"Completed in {result['iteration']} retrieval iteration(s)")
