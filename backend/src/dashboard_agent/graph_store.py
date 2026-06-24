from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
from networkx.readwrite import json_graph

from .s3_source import JsonObject


TOKEN_RE = re.compile(r"[A-Za-z0-9_\-\.]+")


class JsonGraphStore:
    """A persisted, embedded graph index for arbitrary JSON documents."""

    def __init__(self, path: Path):
        self.path = path
        self.graph = nx.DiGraph()
        self.updated_at = 0.0
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.graph = json_graph.node_link_graph(payload, edges="links")
        self.updated_at = float(self.graph.graph.get("updated_at", self.path.stat().st_mtime))

    def rebuild(self, objects: Iterable[JsonObject]) -> dict[str, int]:
        graph = nx.DiGraph()
        object_count = 0
        for source in objects:
            object_count += 1
            root_id = f"object::{source.key}"
            graph.add_node(
                root_id,
                type="dataset",
                label=source.key,
                path=source.key,
                etag=source.etag,
                text=f"S3 JSON dataset {source.key}",
            )
            self._add_value(graph, root_id, source.value, path="$", source_key=source.key)
        self.updated_at = time.time()
        graph.graph.update(updated_at=self.updated_at, object_count=object_count, format="dashboard-agent-json-graph-v1")
        self.graph = graph
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json_graph.node_link_data(graph, edges="links")
        self.path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return {"objects": object_count, "nodes": graph.number_of_nodes(), "edges": graph.number_of_edges()}

    def _add_value(self, graph: nx.DiGraph, parent_id: str, value: Any, *, path: str, source_key: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}"
                child_id = f"{source_key}::{child_path}"
                graph.add_node(child_id, type="field", label=str(key), path=child_path, source=source_key, text=f"{key} {self._preview(child)}")
                graph.add_edge(parent_id, child_id, relation="has_field")
                self._add_value(graph, child_id, child, path=child_path, source_key=source_key)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                child_path = f"{path}[{index}]"
                child_id = f"{source_key}::{child_path}"
                graph.add_node(child_id, type="record", label=f"record {index}", path=child_path, source=source_key, text=self._preview(child))
                graph.add_edge(parent_id, child_id, relation="has_item", index=index)
                self._add_value(graph, child_id, child, path=child_path, source_key=source_key)
        else:
            graph.nodes[parent_id]["value"] = value
            graph.nodes[parent_id]["text"] = f"{graph.nodes[parent_id].get('label', '')} {self._preview(value)}".strip()

    def search(self, question: str, limit: int = 12) -> list[dict[str, Any]]:
        query = self._terms(question)
        scored: list[tuple[float, str, dict[str, Any]]] = []
        for node_id, attrs in self.graph.nodes(data=True):
            text = " ".join(str(attrs.get(key, "")) for key in ("label", "path", "source", "text", "value"))
            terms = self._terms(text)
            overlap = query & terms
            if not overlap:
                continue
            score = len(overlap) / max(len(query), 1)
            if attrs.get("type") == "field":
                score += 0.1
            scored.append((score, str(node_id), attrs))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            {
                "id": node_id,
                "label": attrs.get("label", node_id),
                "path": attrs.get("path"),
                "source": attrs.get("source") or attrs.get("path"),
                "value": attrs.get("value"),
                "text": attrs.get("text", ""),
                "score": round(score, 4),
            }
            for score, node_id, attrs in scored[:limit]
        ]

    def status(self) -> dict[str, Any]:
        return {
            "objects": int(self.graph.graph.get("object_count", 0)),
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "updated_at": self.updated_at or None,
            "path": str(self.path),
        }

    @staticmethod
    def _preview(value: Any, max_length: int = 500) -> str:
        if isinstance(value, (dict, list)):
            text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        else:
            text = str(value)
        return text[:max_length]

    @staticmethod
    def _terms(value: str) -> set[str]:
        return {token.lower() for token in TOKEN_RE.findall(value) if len(token) > 1}
