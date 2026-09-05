"""LangGraph StateGraph — routes to the correct phase node per invocation."""

from langgraph.graph import StateGraph, END

from app.graph.state import InterviewState
from app.graph.nodes import phase1_node, phase2_node, phase3_node


def _route_by_phase(state: dict) -> str:
    """Conditional entry: pick the right phase node based on state."""
    phase = state.get("phase", "open")
    return {
        "open": "phase1",
        "summary": "phase2",
        "deep_dive": "phase3",
    }.get(phase, "__end__")


def build_graph():
    """
    Flow per invocation (one user message):
        route_by_phase → phase1 / phase2 / phase3 → END
    Each phase node handles extract + logic + response internally.
    """
    graph = StateGraph(InterviewState)

    graph.add_node("phase1", phase1_node)
    graph.add_node("phase2", phase2_node)
    graph.add_node("phase3", phase3_node)

    graph.set_conditional_entry_point(_route_by_phase)

    graph.add_edge("phase1", END)
    graph.add_edge("phase2", END)
    graph.add_edge("phase3", END)

    return graph.compile()


# Singleton compiled graph
interview_graph = build_graph()
