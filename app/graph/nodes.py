"""LangGraph nodes — 3 phase nodes + internal helpers. Zero prompt strings here."""

from __future__ import annotations

import json

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import MODEL_NAME
from app.graph.state import CATEGORIES
from app.graph.prompts import (
    INTERVIEWER_SYSTEM,
    EXTRACTOR_SYSTEM,
    PHASE_CHECK_SYSTEM,
    SUMMARY_SYSTEM,
    DEEP_DIVE_SYSTEM,
    CONFIRMATION_CHECK_SYSTEM,
)


# ═══════════════════════════════════════════════════════════════════════
# Pydantic schemas
# ═══════════════════════════════════════════════════════════════════════

class WorkEntry(BaseModel):
    company: str = ""
    role: str = ""
    duration: str = ""
    description: str = ""


class EducationEntry(BaseModel):
    institution: str = ""
    degree: str = ""
    field: str = ""
    year: str = ""


class SkillEntry(BaseModel):
    name: str = ""
    level: str = ""


class ProjectEntry(BaseModel):
    name: str = ""
    description: str = ""
    technologies: list[str] = Field(default_factory=list)
    outcome: str = ""


class ExtractedInfo(BaseModel):
    """All info extractable from a single user message."""
    work_history: list[WorkEntry] = Field(default_factory=list)
    education: list[EducationEntry] = Field(default_factory=list)
    skills: list[SkillEntry] = Field(default_factory=list)
    highlight_project: list[ProjectEntry] = Field(default_factory=list)


class PhaseCheckResult(BaseModel):
    ready: bool
    reason: str = ""


class ConfirmationCheck(BaseModel):
    confirmed: bool
    has_corrections: bool = False


# ═══════════════════════════════════════════════════════════════════════
# LLM instances
# ═══════════════════════════════════════════════════════════════════════

llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
conversational_llm = ChatOpenAI(model=MODEL_NAME, temperature=0.7)

extractor = llm.with_structured_output(ExtractedInfo)
phase_checker = llm.with_structured_output(PhaseCheckResult)
confirmation_checker = llm.with_structured_output(ConfirmationCheck)


# ═══════════════════════════════════════════════════════════════════════
# Helper functions (internal — not graph nodes)
# ═══════════════════════════════════════════════════════════════════════

def _format_collected(collected: dict) -> str:
    """Turn collected dict into a readable summary string for prompts."""
    lines = []
    for cat in CATEGORIES:
        entries = collected.get(cat, [])
        if not entries:
            continue
        label = cat.replace("_", " ").title()
        lines.append(f"\n{label}:")
        for entry in entries:
            if isinstance(entry, dict):
                parts = [f"{k}: {v}" for k, v in entry.items() if v]
                lines.append(f"  - {', '.join(parts)}")
            else:
                lines.append(f"  - {entry}")
    return "\n".join(lines) if lines else "(nothing collected yet)"


def _entry_key(entry: dict, category: str) -> str:
    """Generate a dedup key for an entry based on its identifying fields."""
    if category == "work_history":
        return f"{entry.get('company', '').lower().strip()}|{entry.get('role', '').lower().strip()}"
    elif category == "education":
        return f"{entry.get('institution', '').lower().strip()}|{entry.get('field', '').lower().strip()}"
    elif category == "highlight_project":
        return entry.get("name", "").lower().strip()
    elif category == "skills":
        return entry.get("name", "").lower().strip()
    return json.dumps(entry, sort_keys=True)


def _merge_entry(existing: dict, new: dict) -> dict:
    """Merge new entry into existing — fill empty fields, prefer longer descriptions."""
    merged = dict(existing)
    for k, v in new.items():
        if not v:
            continue
        old_val = merged.get(k, "")
        if not old_val:
            merged[k] = v
        elif isinstance(v, str) and isinstance(old_val, str) and len(v) > len(old_val):
            merged[k] = v
        elif isinstance(v, list) and isinstance(old_val, list):
            merged[k] = list({*old_val, *v})
    return merged


def _extract(state: dict) -> dict:
    """Extract structured info from the latest user message. Returns updated collected."""
    messages = state["messages"]
    if not messages:
        return {"collected": state["collected"]}

    result: ExtractedInfo = extractor.invoke([
        SystemMessage(content=EXTRACTOR_SYSTEM),
        *messages,
    ])

    collected = {cat: list(entries) for cat, entries in state["collected"].items()}
    for cat in CATEGORIES:
        new_entries = getattr(result, cat)
        if not new_entries:
            continue
        # Build index of existing entries by key
        existing_keys = {}
        for i, entry in enumerate(collected[cat]):
            key = _entry_key(entry, cat)
            if key:
                existing_keys[key] = i
        for new_entry in new_entries:
            dumped = new_entry.model_dump()
            key = _entry_key(dumped, cat)
            if key and key in existing_keys:
                # Merge into existing entry (fill gaps, keep longer descriptions)
                idx = existing_keys[key]
                collected[cat][idx] = _merge_entry(collected[cat][idx], dumped)
            else:
                collected[cat].append(dumped)
                if key:
                    existing_keys[key] = len(collected[cat]) - 1

    return {"collected": collected}


def _check_phase_ready(state: dict) -> bool:
    """LLM decides if we have enough big-picture info to move on."""
    # Safety: need at least 3 user messages before we can say "ready"
    user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    if len(user_msgs) < 3:
        return False

    summary = _format_collected(state["collected"])
    result: PhaseCheckResult = phase_checker.invoke([
        SystemMessage(content=PHASE_CHECK_SYSTEM.format(
            collected_summary=summary,
            target_role=state["target_role"],
        )),
    ])
    return result.ready


def _check_confirmation(messages: list) -> ConfirmationCheck:
    """Check if the user's latest message confirms the summary."""
    return confirmation_checker.invoke([
        SystemMessage(content=CONFIRMATION_CHECK_SYSTEM),
        *messages,
    ])


def _build_deep_dive_items(collected: dict) -> list[dict]:
    """Build ordered list of items to deep-dive into. Skip skills."""
    items = []
    for cat in ["work_history", "education", "highlight_project"]:
        for i, entry in enumerate(collected.get(cat, [])):
            if isinstance(entry, dict):
                # Build a readable label
                if cat == "work_history":
                    label = f"{entry.get('role', '?')} at {entry.get('company', '?')}"
                elif cat == "education":
                    label = f"{entry.get('degree', '?')} at {entry.get('institution', '?')}"
                else:
                    label = entry.get("name", "Project")
            else:
                label = str(entry)
            items.append({"category": cat, "index": i, "label": label})
    return items


def _generate_summary(collected: dict, messages: list) -> str:
    """Generate a human-friendly summary of collected info."""
    summary = _format_collected(collected)
    response = conversational_llm.invoke([
        SystemMessage(content=SUMMARY_SYSTEM.format(collected_summary=summary)),
        *messages,
    ])
    return response.content


def _generate_deep_dive_question(messages: list, item: dict, collected: dict) -> str:
    """Generate the opening deep-dive question for an item."""
    entry = collected[item["category"]][item["index"]]
    item_details = json.dumps(entry, indent=2, ensure_ascii=False)
    response = conversational_llm.invoke([
        SystemMessage(content=DEEP_DIVE_SYSTEM.format(
            current_item=item["label"],
            item_details=item_details,
        )),
        *messages,
    ])
    return response.content


def generate_opening(target_role: str) -> str:
    """Generate the opening question for a new interview. Used by main.py."""
    response = conversational_llm.invoke([
        SystemMessage(content=INTERVIEWER_SYSTEM.format(
            target_role=target_role,
            collected_summary="(nothing yet — this is the very first message)",
        )),
    ])
    return response.content


# ═══════════════════════════════════════════════════════════════════════
# Graph nodes — one per phase
# ═══════════════════════════════════════════════════════════════════════

def phase1_node(state: dict) -> dict:
    """Phase 1: extract info, check if ready, generate conversational follow-up."""
    # 1. Extract
    updates = _extract(state)
    merged_collected = updates["collected"]

    # 2. Check if ready for Phase 2
    check_state = {**state, "collected": merged_collected}
    if _check_phase_ready(check_state):
        summary = _generate_summary(merged_collected, state["messages"])
        return {
            "collected": merged_collected,
            "phase": "summary",
            "current_question": summary,
        }

    # 3. Generate conversational follow-up
    collected_summary = _format_collected(merged_collected)
    response = conversational_llm.invoke([
        SystemMessage(content=INTERVIEWER_SYSTEM.format(
            target_role=state["target_role"],
            collected_summary=collected_summary,
        )),
        *state["messages"],
    ])
    return {
        "collected": merged_collected,
        "current_question": response.content,
    }


def phase2_node(state: dict) -> dict:
    """Phase 2: extract corrections, check confirmation, re-summarize or transition."""
    # 1. Extract (in case user added corrections)
    updates = _extract(state)
    merged_collected = updates["collected"]

    # 2. Check if user confirmed
    confirmation = _check_confirmation(state["messages"])

    if confirmation.confirmed and not confirmation.has_corrections:
        # Build deep-dive items and transition to Phase 3
        items = _build_deep_dive_items(merged_collected)
        if not items:
            return {
                "collected": merged_collected,
                "phase": "done",
                "finished": True,
                "current_question": "Great, I think I have everything I need! Thank you.",
            }
        first_q = _generate_deep_dive_question(state["messages"], items[0], merged_collected)
        return {
            "collected": merged_collected,
            "phase": "deep_dive",
            "summary_confirmed": True,
            "deep_dive_items": items,
            "deep_dive_cursor": 0,
            "current_question": first_q,
        }

    # 3. Re-summarize with updated data
    summary = _generate_summary(merged_collected, state["messages"])
    return {
        "collected": merged_collected,
        "current_question": summary,
    }


def phase3_node(state: dict) -> dict:
    """Phase 3: deep-dive questions, advance items, finish."""
    # 1. Extract detailed info
    updates = _extract(state)
    merged_collected = updates["collected"]

    # 2. Get current item
    items = state["deep_dive_items"]
    cursor = state["deep_dive_cursor"]
    current = items[cursor]

    # 3. Generate deep-dive response
    entry = merged_collected[current["category"]][current["index"]]
    item_details = json.dumps(entry, indent=2, ensure_ascii=False)
    response = conversational_llm.invoke([
        SystemMessage(content=DEEP_DIVE_SYSTEM.format(
            current_item=current["label"],
            item_details=item_details,
        )),
        *state["messages"],
    ])

    reply = response.content

    # 4. Check for DONE signal (must be last line, exactly "DONE")
    lines = reply.strip().split("\n")
    if lines[-1].strip() == "DONE":
        reply = "\n".join(lines[:-1]).strip()
        next_cursor = cursor + 1

        if next_cursor >= len(items):
            closing = reply + "\n\nThat covers everything! Thanks for sharing all of this."
            return {
                "collected": merged_collected,
                "deep_dive_cursor": next_cursor,
                "phase": "done",
                "finished": True,
                "current_question": closing,
            }

        # Move to next item
        next_item = items[next_cursor]
        next_q = _generate_deep_dive_question(state["messages"], next_item, merged_collected)
        transition = f"{reply}\n\n{next_q}" if reply else next_q
        return {
            "collected": merged_collected,
            "deep_dive_cursor": next_cursor,
            "current_question": transition,
        }

    return {
        "collected": merged_collected,
        "current_question": reply,
    }
