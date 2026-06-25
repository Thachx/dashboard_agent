from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from dashboard_agent.graph_store import JsonGraphStore
from dashboard_agent.s3_source import JsonObject, S3JsonSource


def ingest_s3_with_graphify(
    source: S3JsonSource,
    store: JsonGraphStore,
    *,
    graphify_output_dir: Path,
    graphify_bin: str = "graphify",
    graphify_enabled: bool = True,
) -> dict[str, Any]:
    """Ingest S3 content, preferring Graphify graph.json as the searchable graph."""

    objects = source.load()
    if graphify_enabled:
        objects = list(objects)
    stats: dict[str, Any] = store.rebuild(objects)
    stats["s3_uri"] = source.uri
    stats["graphify_enabled"] = graphify_enabled

    if not graphify_enabled:
        stats["graphify_status"] = "disabled"
        return stats
    if not objects:
        stats["graphify_status"] = "skipped-empty-source"
        return stats

    try:
        graphify_stats = _run_graphify(objects, graphify_output_dir, graphify_bin)
        graphify_graph_path = Path(graphify_stats["graphify_path"])
        stats = store.replace_from_graph_json(graphify_graph_path, graph_kind="graphify")
    except Exception as exc:
        stats["graphify_status"] = "failed"
        stats["graphify_error"] = str(exc)
    else:
        stats.update(graphify_stats)
        stats["graphify_status"] = "ok"

    stats["s3_uri"] = source.uri
    stats["graphify_enabled"] = graphify_enabled
    return stats


def _run_graphify(objects: Iterable[JsonObject], output_dir: Path, graphify_bin: str) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="dashboard-agent-s3-") as tmp:
        workspace = Path(tmp)
        corpus_dir = workspace / "s3-corpus"
        _write_corpus(objects, corpus_dir)

        command_prefix = _graphify_command(graphify_bin)
        result = _run_command(
            command_prefix + ["extract", str(corpus_dir), "--out", str(output_dir), "--no-cluster"],
            cwd=output_dir,
        )
        if result.returncode != 0 and _is_missing_semantic_api_key(result):
            result = _run_command(
                command_prefix + ["update", str(corpus_dir), "--no-cluster", "--force"],
                cwd=output_dir,
            )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(f"graphify exited with {result.returncode}: {detail}")

        graph_json = _find_graphify_graph(output_dir, corpus_dir)
        if graph_json is None:
            raise RuntimeError("graphify completed but did not produce graph.json")

        target = output_dir / "graph.json"
        if graph_json.resolve() != target.resolve():
            shutil.copyfile(graph_json, target)
        report = graph_json.parent / "GRAPH_REPORT.md"
        if report.exists():
            shutil.copyfile(report, output_dir / "GRAPH_REPORT.md")

    return {"graphify_path": str(output_dir / "graph.json")}


def _run_command(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )


def _write_corpus(objects: Iterable[JsonObject], corpus_dir: Path) -> None:
    for source in objects:
        path = corpus_dir / _safe_object_path(source.key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(source.value, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_object_path(key: str) -> Path:
    clean_parts = []
    for part in PurePosixPath(key).parts:
        if part in {"", ".", ".."}:
            continue
        clean_parts.append(part.replace(":", "_"))
    if not clean_parts:
        clean_parts = ["object.json"]
    path = Path(*clean_parts)
    if path.suffix.lower() != ".json":
        path = Path(f"{path}.metadata.json")
    return path


def _graphify_command(graphify_bin: str) -> list[str]:
    if Path(graphify_bin).exists() or shutil.which(graphify_bin):
        return [graphify_bin]
    return [sys.executable, "-m", "graphify"]


def _find_graphify_graph(*roots: Path) -> Path | None:
    candidates: list[Path] = []
    for root in roots:
        if root.exists():
            candidates.extend(root.rglob("graph.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _is_missing_semantic_api_key(result: subprocess.CompletedProcess[str]) -> bool:
    output = f"{result.stderr}\n{result.stdout}".lower()
    return "no llm api key found" in output and "semantic extraction" in output
