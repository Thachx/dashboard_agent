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
HUMAN_LABEL_RE = re.compile(r"([a-z0-9])([A-Z])")
HUMAN_KEEP_ALL_CAPS = {"API", "CSV", "DB", "ETAG", "ID", "JSON", "LLM", "SQL", "S3", "UI", "URL", "UTC"}
MAX_RUNTIME_GRAPH_NODES = 5_000
MAX_RUNTIME_GRAPH_EDGES = 10_000
MAX_LOAD_GRAPH_BYTES = 50 * 1024 * 1024


class JsonGraphStore:
    """Persisted graph index backed by Graphify graph.json when available."""

    def __init__(self, path: Path, *, load_existing: bool = True):
        self.path = path
        self.graph = nx.DiGraph()
        self.search_rows: list[dict[str, Any]] = []
        self.index_status: dict[str, Any] = {}
        self.updated_at = 0.0
        if load_existing:
            self.load()

    def load(self) -> None:
        if not self.path.exists():
            self._load_search_index()
            return
        index_path = self._index_path()
        if index_path.exists() and index_path.stat().st_mtime >= self.path.stat().st_mtime:
            self._load_search_index()
            self.updated_at = float(self.index_status.get("updated_at") or self.path.stat().st_mtime)
            return
        if self.path.stat().st_size > MAX_LOAD_GRAPH_BYTES:
            if self._load_search_index():
                self.graph.graph.update(self.index_status)
                self.updated_at = float(self.index_status.get("updated_at") or self.path.stat().st_mtime)
            else:
                self.graph.graph.update(
                    graph_kind="oversized",
                    skipped_graph_path=str(self.path),
                    skipped_graph_bytes=self.path.stat().st_size,
                )
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.graph = self._node_link_graph(payload)
        self.updated_at = float(self.graph.graph.get("updated_at", self.path.stat().st_mtime))
        if not self._load_search_index():
            self._write_search_index()

    def replace_from_graph_json(self, graph_path: Path, *, graph_kind: str = "graphify") -> dict[str, Any]:
        payload = json.loads(graph_path.read_text(encoding="utf-8"))
        graph = self._node_link_graph(payload)
        if graph.number_of_nodes() == 0:
            raise ValueError(f"{graph_kind} graph has no nodes: {graph_path}")
        if graph.number_of_nodes() > MAX_RUNTIME_GRAPH_NODES:
            graph = self._sample_graph(graph, max_nodes=MAX_RUNTIME_GRAPH_NODES, max_edges=MAX_RUNTIME_GRAPH_EDGES)
        self.updated_at = time.time()
        graph.graph.update(
            updated_at=self.updated_at,
            object_count=int(graph.graph.get("object_count", graph.number_of_nodes())),
            format=f"dashboard-agent-{graph_kind}-graph-v1",
            graph_kind=graph_kind,
            source_graph_path=str(graph_path),
        )
        self.graph = graph
        self._persist()
        return self.status()

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
                ETAG=source.ETAG,
                object_type=source.object_type,
                size=source.size,
                text=f"S3 {source.object_type} dataset {source.key}",
            )
            if source.object_type == "duckdb_aggregate":
                graph.nodes[root_id]["type"] = "aggregate"
                graph.nodes[root_id]["value"] = source.value
                graph.nodes[root_id]["text"] = f"DuckDB aggregate {source.key} {self._preview(source.value, 4000)}"
            self._add_value(graph, root_id, source.value, path="$", source_key=source.key)

        self.updated_at = time.time()
        graph.graph.update(
            updated_at=self.updated_at,
            object_count=object_count,
            format="dashboard-agent-json-graph-v1",
            graph_kind="fallback",
        )
        self.graph = graph
        self._persist()
        return self.status()

    def _add_value(self, graph: nx.DiGraph, parent_id: str, value: Any, *, path: str, source_key: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}"
                child_id = f"{source_key}::{child_path}"
                graph.add_node(
                    child_id,
                    type="field",
                    label=str(key),
                    path=child_path,
                    source=source_key,
                    text=f"{key} {self._preview(child)}",
                )
                graph.add_edge(parent_id, child_id, relation="has_field")
                self._add_value(graph, child_id, child, path=child_path, source_key=source_key)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                child_path = f"{path}[{index}]"
                child_id = f"{source_key}::{child_path}"
                graph.add_node(
                    child_id,
                    type="record",
                    label=f"[{index}]",
                    path=child_path,
                    source=source_key,
                    text=self._preview(child),
                )
                graph.add_edge(parent_id, child_id, relation="has_item")
                self._add_value(graph, child_id, child, path=child_path, source_key=source_key)
        else:
            graph.nodes[parent_id]["value"] = value
            graph.nodes[parent_id]["text"] = f"{graph.nodes[parent_id].get('text', '')} {value}".strip()

    def search(self, text: str, limit: int = 8) -> list[dict[str, Any]]:
        query = self._terms(text)
        if not query:
            return []
        direct = self._direct_matches(text, query, limit)
        if direct:
            return direct
        if self.search_rows:
            return self._search_rows(query, limit)
        broad_question = bool(query & {"dashboard", "summary", "overview", "project", "data", "dataset", "graph"})
        scored: list[tuple[float, str, dict[str, Any]]] = []
        started_at = time.perf_counter()
        for index, (node_id, attrs) in enumerate(self.graph.nodes(data=True)):
            if index >= 100_000 or time.perf_counter() - started_at > 1.5:
                break
            node_text = self._node_text(node_id, attrs)
            terms = self._terms(node_text)
            overlap = query & terms
            if not overlap:
                continue
            score = len(overlap) / max(len(query), 1)
            node_type = str(attrs.get("type") or attrs.get("file_type") or "").lower()
            path = str(attrs.get("path") or attrs.get("file") or attrs.get("source") or "").lower().replace("\\", "/")
            label = str(attrs.get("label") or attrs.get("name") or "").lower()
            if node_type in {"field", "column", "dataset"}:
                score += 0.1
            if broad_question and node_type in {"dataset", "repository"}:
                score += 0.25
            if broad_question and (label.startswith("readme") or "/readme" in path):
                score += 0.15
            lowered_text = node_text.lower()
            for query_term in query:
                if query_term in lowered_text:
                    score += 0.05
            scored.append((score, str(node_id), attrs))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [self._result_item(node_id, attrs, score) for score, node_id, attrs in scored[:limit]]

    def _direct_matches(self, text: str, query: set[str], limit: int) -> list[dict[str, Any]]:
        matches: list[tuple[float, str, dict[str, Any]]] = []
        raw_tokens = [token.lower() for token in TOKEN_RE.findall(text) if len(token) > 1]
        candidates = set(raw_tokens)
        for token in raw_tokens:
            if "." not in token:
                candidates.add(f"{token}.json")
        if self.search_rows:
            row_by_id = {str(row.get("id", "")).lower(): row for row in self.search_rows}
            row_by_label = {str(row.get("label", "")).lower(): row for row in self.search_rows}
            direct_rows = []
            for candidate in candidates:
                row = row_by_id.get(candidate) or row_by_label.get(candidate)
                if row:
                    direct_rows.append(self._row_result(row, 1.5))
            if direct_rows:
                return direct_rows[:limit]
        for candidate in candidates:
            if candidate in self.graph:
                attrs = self.graph.nodes[candidate]
                matches.append((1.5, candidate, attrs))

        if not matches:
            joined = "_".join(part for part in raw_tokens if part not in {"show", "dashboard", "graph", "widget", "for"})
            if joined in self.graph:
                attrs = self.graph.nodes[joined]
                matches.append((1.25, joined, attrs))

        return [self._result_item(node_id, attrs, score) for score, node_id, attrs in matches[:limit]]

    def _search_rows(self, query: set[str], limit: int) -> list[dict[str, Any]]:
        scored: list[tuple[float, dict[str, Any]]] = []
        started_at = time.perf_counter()
        for index, row in enumerate(self.search_rows):
            if index >= 200_000 or time.perf_counter() - started_at > 0.75:
                break
            terms = set(row.get("terms") or [])
            overlap = query & terms
            if not overlap:
                continue
            score = len(overlap) / max(len(query), 1)
            row_text = str(row.get("text") or "").lower()
            for query_term in query:
                if query_term in row_text:
                    score += 0.05
            scored.append((score, row))
        scored.sort(key=lambda item: (-item[0], str(item[1].get("id", ""))))
        return [self._row_result(row, score) for score, row in scored[:limit]]

    def _result_item(self, node_id: str, attrs: dict[str, Any], score: float) -> dict[str, Any]:
        return {
            "id": node_id,
            "label": self._human_label(attrs.get("label") or attrs.get("name") or node_id),
            "path": attrs.get("path") or attrs.get("file") or attrs.get("source_file") or attrs.get("source"),
            "source": attrs.get("source") or attrs.get("source_file") or attrs.get("path") or attrs.get("file"),
            "value": attrs.get("value"),
            "text": attrs.get("text") or attrs.get("summary") or attrs.get("content") or self._node_text(node_id, attrs),
            "score": round(score, 4),
        }

    def neighbors(self, node_id: str, limit: int = 5) -> list[str]:
        if node_id not in self.graph:
            return []
        labels: list[str] = []
        for neighbor in list(self.graph.neighbors(node_id))[:limit]:
            attrs = self.graph.nodes[neighbor]
            labels.append(self._human_label(attrs.get("label") or attrs.get("name") or neighbor))
        return labels

    def status(self) -> dict[str, Any]:
        if self.index_status and not self.graph.number_of_nodes():
            status = dict(self.index_status)
            status.setdefault("path", str(self.path))
            return status
        return {
            "objects": int(self.graph.graph.get("object_count", 0)),
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "updated_at": self.updated_at or None,
            "path": str(self.path),
            "graph_kind": self.graph.graph.get("graph_kind") or "app",
            "sampled": bool(self.graph.graph.get("sampled", False)),
            "full_nodes": self.graph.graph.get("full_node_count"),
            "full_edges": self.graph.graph.get("full_edge_count"),
        }

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json_graph.node_link_data(self.graph, edges="links")
        self.path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        self._write_search_index()

    def _index_path(self) -> Path:
        return self.path.with_name(f"{self.path.name}.search.json")

    def _load_search_index(self) -> bool:
        index_path = self._index_path()
        if not index_path.exists():
            return False
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        self.index_status = dict(payload.get("status") or {})
        self.search_rows = list(payload.get("rows") or [])
        return True

    def _write_search_index(self) -> None:
        rows: list[dict[str, Any]] = []
        for node_id, attrs in self.graph.nodes(data=True):
            row = self._result_item(str(node_id), attrs, 0)
            row["terms"] = sorted(self._terms(" ".join(str(row.get(key) or "") for key in ("id", "label", "path", "source", "text"))))
            rows.append(row)
        self.search_rows = rows
        self.index_status = self.status()
        payload = {"version": 1, "status": self.index_status, "rows": rows}
        self._index_path().write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _row_result(row: dict[str, Any], score: float) -> dict[str, Any]:
        return {
            "id": row.get("id"),
            "label": row.get("label"),
            "path": row.get("path"),
            "source": row.get("source"),
            "value": row.get("value"),
            "text": row.get("text") or "",
            "score": round(score, 4),
        }

    @staticmethod
    def _node_link_graph(payload: dict[str, Any]) -> nx.Graph:
        if isinstance(payload, dict) and "nodes" in payload and "links" in payload:
            return json_graph.node_link_graph(payload, edges="links")
        if isinstance(payload, dict) and "nodes" in payload and "edges" in payload:
            return json_graph.node_link_graph(payload, edges="edges")
        raise ValueError("Unsupported graph JSON format.")

    @staticmethod
    def _sample_graph(graph: nx.Graph, *, max_nodes: int, max_edges: int) -> nx.Graph:
        ordered_nodes = sorted(
            graph.nodes,
            key=lambda node_id: (-graph.degree[node_id], str(graph.nodes[node_id].get("label") or node_id).lower()),
        )
        sampled_nodes = set(ordered_nodes[:max_nodes])
        sampled = graph.__class__()
        for node_id in ordered_nodes[:max_nodes]:
            sampled.add_node(node_id, **graph.nodes[node_id])
        edge_count = 0
        for source, target, attrs in graph.edges(data=True):
            if source not in sampled_nodes or target not in sampled_nodes:
                continue
            sampled.add_edge(source, target, **attrs)
            edge_count += 1
            if edge_count >= max_edges:
                break
        sampled.graph.update(graph.graph)
        sampled.graph["sampled"] = True
        sampled.graph["full_node_count"] = graph.number_of_nodes()
        sampled.graph["full_edge_count"] = graph.number_of_edges()
        return sampled

    @staticmethod
    def _node_text(node_id: object, attrs: dict[str, Any]) -> str:
        parts = [str(node_id)]
        for key in (
            "label",
            "name",
            "type",
            "file_type",
            "path",
            "file",
            "source",
            "source_file",
            "source_location",
            "summary",
            "text",
            "content",
            "value",
        ):
            value = attrs.get(key)
            if value is not None and value != "":
                parts.append(str(value))
        return " | ".join(parts)

    @staticmethod
    def _preview(value: Any, max_length: int = 500) -> str:
        if isinstance(value, (dict, list)):
            text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        else:
            text = str(value)
        return text[:max_length]

    @staticmethod
    def _terms(value: str) -> set[str]:
        terms: set[str] = set()
        for token in TOKEN_RE.findall(value):
            term = token.lower()
            if len(term) <= 1:
                continue
            terms.add(term)
            for part in re.split(r"[_\-.]+", term):
                if len(part) > 1:
                    terms.add(part)
            if len(term) > 3 and term.endswith("s"):
                terms.add(term[:-1])
        return terms

    @staticmethod
    def _human_label(value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return "Unknown"
        text = text.replace("\\", "/").rsplit("/", 1)[-1]
        text = re.sub(r"^\$\.?", "", text)
        text = re.sub(r"\.(json|csv|tsv|parquet|ndjson|jsonl|sql|db|duckdb|txt)$", "", text, flags=re.IGNORECASE)
        text = HUMAN_LABEL_RE.sub(r"\1 \2", text)
        text = re.sub(r"[_\-.]+", " ", text)
        words: list[str] = []
        for raw_word in text.split():
            word = raw_word.strip()
            if not word:
                continue
            if word.upper() in HUMAN_KEEP_ALL_CAPS:
                words.append(word.upper() if len(word) <= 4 else word.title())
                continue
            if word.isupper() and len(word) <= 4:
                words.append(word)
                continue
            if word.isdigit():
                words.append(word)
                continue
            words.append(word[:1].upper() + word[1:].lower())
        return " ".join(words) if words else "Unknown"
