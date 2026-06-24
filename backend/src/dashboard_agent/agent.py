from __future__ import annotations

import json
import threading
import time
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from dashboard_agent.config import Settings
from dashboard_agent.graph_store import JsonGraphStore
from dashboard_agent.powerbi import PowerBiEmbed
from dashboard_agent.s3_source import S3JsonSource


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    context: dict[str, Any]


settings = Settings.from_env()
store = JsonGraphStore(settings.graph_path)
refresh_lock = threading.Lock()


def refresh_graph(force: bool = False) -> dict[str, Any]:
    age = time.time() - store.updated_at if store.updated_at else float("inf")
    if not force and store.graph.number_of_nodes() and age < settings.graph_refresh_seconds:
        return store.status()
    with refresh_lock:
        age = time.time() - store.updated_at if store.updated_at else float("inf")
        if not force and store.graph.number_of_nodes() and age < settings.graph_refresh_seconds:
            return store.status()
        source = S3JsonSource(
            settings.s3_data_uri,
            region=settings.aws_region,
            max_object_bytes=settings.graph_max_object_bytes,
        )
        return store.rebuild(source.load())


def _last_question(state: AgentState) -> str:
    for message in reversed(state.get("messages", [])):
        if getattr(message, "type", "") == "human":
            content = getattr(message, "content", "")
            return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    return ""


def _wants_dashboard(question: str) -> bool:
    lowered = question.lower()
    return "power bi" in lowered or "powerbi" in lowered or "dashboard" in lowered


def _wants_refresh(question: str) -> bool:
    lowered = question.lower()
    return any(term in lowered for term in ("refresh", "reload", "sync", "reindex", "ingest"))


def _fallback_answer(question: str, results: list[dict[str, Any]]) -> str:
    if not results:
        return "I could not find matching values in the indexed S3 JSON graph. Try naming a dataset, field, or value."
    lines = ["I found these relevant values in the S3 JSON graph:"]
    for item in results[:8]:
        value = item.get("value")
        detail = str(value) if value is not None else str(item.get("text", ""))
        lines.append(f"- `{item.get('source')}` / `{item.get('path')}`: {detail[:240]}")
    return "\n".join(lines)


def _llm_answer(question: str, results: list[dict[str, Any]]) -> str:
    if not settings.openai_api_key:
        return _fallback_answer(question, results)
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        base_url=settings.openai_base_url,
        temperature=0,
    )
    context = json.dumps(results, ensure_ascii=False, default=str)
    response = model.invoke(
        [
            ("system", "Answer using only the supplied S3 JSON graph context. Cite JSON paths in backticks. State when evidence is insufficient."),
            ("human", f"Question: {question}\n\nGraph context:\n{context}"),
        ]
    )
    return str(response.content)


def run_agent(state: AgentState) -> dict[str, list[AIMessage]]:
    question = _last_question(state)
    notices: list[str] = []
    try:
        status = refresh_graph(force=_wants_refresh(question))
        if _wants_refresh(question):
            notices.append(f"S3 graph refreshed: {status['objects']} objects, {status['nodes']} nodes, {status['edges']} edges.")
    except Exception as exc:
        if not store.graph.number_of_nodes():
            notices.append(f"S3 ingestion is unavailable: {exc}")
        else:
            notices.append(f"Using the last saved graph because S3 refresh failed: {exc}")

    results = store.search(question)
    answer = _llm_answer(question, results)
    if _wants_dashboard(question):
        embed = PowerBiEmbed(settings.power_bi_embed_url, settings.power_bi_report_id, settings.power_bi_access_token)
        answer = f"{answer}\n\n{embed.marker()}"
    if notices:
        answer = "\n\n".join(notices + [answer])
    return {"messages": [AIMessage(content=answer)]}


builder = StateGraph(AgentState)
builder.add_node("dashboard_agent", run_agent)
builder.add_edge(START, "dashboard_agent")
builder.add_edge("dashboard_agent", END)
graph = builder.compile()
