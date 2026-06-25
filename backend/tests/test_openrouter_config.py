from pathlib import Path

from dashboard_agent import agent
from dashboard_agent.config import Settings


def test_openrouter_candidates_try_main_then_two_reserves(monkeypatch):
    monkeypatch.setattr(
        agent,
        "settings",
        Settings(
            graph_path=Path("unused.json"),
            openrouter_api_key="or-key",
            openrouter_main_model="anthropic/claude-main",
            openrouter_reserve_model_1="openai/gpt-reserve",
            openrouter_reserve_model_2="google/gemini-reserve",
            openrouter_http_referer="https://example.com",
            openrouter_title="Dashboard Graph Agent",
        ),
    )

    candidates = agent._llm_candidates()

    assert [candidate["model"] for candidate in candidates] == [
        "anthropic/claude-main",
        "openai/gpt-reserve",
        "google/gemini-reserve",
    ]
    assert all(candidate["provider"] == "openrouter" for candidate in candidates)
    assert candidates[0]["base_url"] == "https://openrouter.ai/api/v1"
    assert candidates[0]["headers"] == {
        "HTTP-Referer": "https://example.com",
        "X-OpenRouter-Title": "Dashboard Graph Agent",
    }


def test_openrouter_duplicate_reserve_models_are_skipped(monkeypatch):
    monkeypatch.setattr(
        agent,
        "settings",
        Settings(
            graph_path=Path("unused.json"),
            openrouter_api_key="or-key",
            openrouter_main_model="openai/gpt-4.1-mini",
            openrouter_reserve_model_1="openai/gpt-4.1-mini",
            openrouter_reserve_model_2="meta-llama/llama-3.3-70b-instruct",
        ),
    )

    assert [candidate["model"] for candidate in agent._llm_candidates()] == [
        "openai/gpt-4.1-mini",
        "meta-llama/llama-3.3-70b-instruct",
    ]


def test_llm_mode_never_disables_llm(monkeypatch):
    monkeypatch.setattr(
        agent,
        "settings",
        Settings(graph_path=Path("unused.json"), llm_mode="never", openrouter_api_key="or-key"),
    )

    assert agent._should_use_llm([{"path": "$.metric"}]) is False


def test_llm_mode_auto_requires_graph_context(monkeypatch):
    monkeypatch.setattr(
        agent,
        "settings",
        Settings(graph_path=Path("unused.json"), llm_mode="auto", openrouter_api_key="or-key"),
    )

    assert agent._should_use_llm([]) is False
    assert agent._should_use_llm([{"path": "$.metric"}]) is True


def test_llm_mode_always_uses_llm_without_graph_context(monkeypatch):
    monkeypatch.setattr(
        agent,
        "settings",
        Settings(graph_path=Path("unused.json"), llm_mode="always", openrouter_api_key="or-key"),
    )

    assert agent._should_use_llm([]) is True


def test_llm_provider_openai_uses_only_openai(monkeypatch):
    monkeypatch.setattr(
        agent,
        "settings",
        Settings(
            graph_path=Path("unused.json"),
            llm_provider="openai",
            openrouter_api_key="or-key",
            openai_api_key="openai-key",
            openai_model="gpt-test",
        ),
    )

    candidates = agent._llm_candidates()

    assert [candidate["provider"] for candidate in candidates] == ["openai"]
    assert candidates[0]["model"] == "gpt-test"


def test_llm_provider_auto_uses_openrouter_then_openai(monkeypatch):
    monkeypatch.setattr(
        agent,
        "settings",
        Settings(
            graph_path=Path("unused.json"),
            llm_provider="auto",
            openrouter_api_key="or-key",
            openrouter_main_model="or-main",
            openai_api_key="openai-key",
            openai_model="gpt-test",
        ),
    )

    assert [(candidate["provider"], candidate["model"]) for candidate in agent._llm_candidates()] == [
        ("openrouter", "or-main"),
        ("openai", "gpt-test"),
    ]
