"""Interview state definition — the central data structure for each interview session."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


# ── Collected info schema ────────────────────────────────────────────
# Each category maps to a list of entries extracted from the conversation.

CATEGORIES = ["work_history", "education", "skills", "highlight_project"]


def _empty_collected() -> dict[str, list]:
    return {cat: [] for cat in CATEGORIES}


# ── LangGraph state ─────────────────────────────────────────────────

class InterviewState(TypedDict):
    session_id: str
    target_role: str
    # LangGraph's add_messages reducer: appends new messages instead of replacing
    messages: Annotated[list, add_messages]
    collected: dict               # category -> list of extracted entries
    current_question: str         # last question sent to the user
    finished: bool                # True when interview is complete

    # ── Phase tracking ───────────────────────────────────────────────
    phase: str                    # "open" | "summary" | "deep_dive" | "done"
    deep_dive_items: list         # ordered list of items to deep-dive into
    deep_dive_cursor: int         # index into deep_dive_items
    summary_confirmed: bool       # True after user confirms the summary
