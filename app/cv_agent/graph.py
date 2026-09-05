"""CV Agent LangGraph — connects all nodes into a pipeline."""

from langgraph.graph import StateGraph, END

from app.cv_agent.state import CVAgentState
from app.cv_agent.nodes import (
    router_node,
    retrieve_node,
    evaluate_node,
    generate_node,
    verify_node,
)


def _should_retrieve_more(state: dict) -> str:
    """After evaluate: go back to retrieve or forward to generate."""
    if state.get("needs_more_data"):
        return "retrieve"
    return "generate"


def build_cv_graph():
    """
    Flow:
        router → retrieve → evaluate →
            if needs more data → retrieve (loop, max 3 times)
            if sufficient → generate → verify → END
    """
    graph = StateGraph(CVAgentState)

    # Add nodes
    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("generate", generate_node)
    graph.add_node("verify", verify_node)

    # Connect edges
    graph.set_entry_point("router")
    graph.add_edge("router", "retrieve")
    graph.add_edge("retrieve", "evaluate")

    # Conditional: evaluate decides if we need more data or can generate
    graph.add_conditional_edges("evaluate", _should_retrieve_more, {
        "retrieve": "retrieve",
        "generate": "generate",
    })

    graph.add_edge("generate", "verify")
    graph.add_edge("verify", END)

    return graph.compile()


# Ready-to-use compiled graph
cv_agent = build_cv_graph()
