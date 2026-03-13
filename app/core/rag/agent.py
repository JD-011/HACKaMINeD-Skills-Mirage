"""
LangGraph Agentic Workflow
--------------------------
An intelligent ReAct agent that decides at runtime which tools to invoke
(Layer-1 PostgreSQL data and/or Qdrant RAG course search) based on what the
user is actually asking.

Uses langgraph-prebuilt's `create_react_agent` for a standard tool-calling loop.
"""

import json
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from app.config import settings
from app.core.rag.tools import get_all_tools
from app.core.rag.prompts import build_agent_system_prompt
from app.models.schemas import UserProfile


def _get_llm() -> ChatGroq:
    """Return a Groq-hosted Llama 3.3 70B chat model with tool-calling support."""
    return ChatGroq(
        model=settings.LLM_MODEL,
        api_key=settings.GROQ_API_KEY,
        temperature=0.3,
    )


async def run_agent(query: str, user: UserProfile, memory_context: str = "") -> dict:
    """
    Run the LangGraph agentic workflow.

    1. Build a system prompt personalised to the user's profile + memory
    2. Create a ReAct agent armed with Layer-1 DB tools + RAG course search
    3. Execute the agent loop (the LLM decides which tools to call)
    4. Extract final answer, sources, and list of tools that were invoked

    Returns
    -------
    dict  {"answer": str, "sources": list[dict], "tools_used": list[str]}
    """
    system_prompt = build_agent_system_prompt(user, memory_context)

    llm = _get_llm()
    tools = get_all_tools()

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=query)]}
    )

    messages = result["messages"]

    # ── Extract which tools were called ──────────────────────────────
    tools_used: set[str] = set()
    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                tools_used.add(tc["name"])

    # ── Extract final answer (last AIMessage with content, no pending tool calls)
    final_answer = ""
    for msg in reversed(messages):
        if (
            isinstance(msg, AIMessage)
            and msg.content
            and not getattr(msg, "tool_calls", None)
        ):
            final_answer = msg.content
            break

    # ── Extract course sources from tool results ─────────────────────
    sources: list[dict] = []
    seen_titles: set[str] = set()
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.name == "search_courses":
            try:
                courses = json.loads(msg.content)
                # Could be a list or a dict with "courses" key
                if isinstance(courses, dict):
                    courses = courses.get("courses", [])
                for c in courses:
                    if isinstance(c, dict):
                        title = c.get("title", "")
                        if title and title not in seen_titles:
                            seen_titles.add(title)
                            sources.append({
                                "title": title,
                                "platform": c.get("platform", ""),
                                "link": c.get("link", ""),
                                "institute": c.get("institute", ""),
                            })
            except (json.JSONDecodeError, TypeError):
                pass

    return {
        "answer": final_answer,
        "sources": sources,
        "tools_used": list(tools_used),
    }
