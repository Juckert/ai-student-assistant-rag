"""
Tests for agent/agent.py.

Ollama is fully mocked — no real HTTP connection is made.
Covers:
  - message is sent to the model (HTTP POST is called)
  - response text is returned to the caller
  - question text appears inside the prompt payload
  - empty context short-circuits without calling Ollama
"""
from unittest.mock import patch, MagicMock

import pytest

from agent.agent import generate_answer


# ── helpers ────────────────────────────────────────────────────────────────

def _ok_response(text: str) -> MagicMock:
    mock = MagicMock()
    mock.json.return_value = {"response": text}
    mock.raise_for_status.return_value = None
    return mock


# ── tests ──────────────────────────────────────────────────────────────────

def test_message_is_sent_to_ollama():
    """A question with non-empty context must trigger exactly one POST request."""
    with patch("agent.agent.requests.post") as mock_post:
        mock_post.return_value = _ok_response("Ответ.")
        generate_answer("Что такое Python?", ["Python — язык программирования."])
        assert mock_post.call_count == 1


def test_response_text_is_returned():
    """The exact text from Ollama's JSON payload is returned to the caller."""
    expected = "Python — высокоуровневый язык программирования."
    with patch("agent.agent.requests.post") as mock_post:
        mock_post.return_value = _ok_response(expected)
        result = generate_answer("Что такое Python?", ["Python — язык программирования."])
    assert result == expected


def test_question_is_included_in_prompt():
    """The user's question must appear verbatim inside the prompt sent to Ollama."""
    question = "Что такое нейронная сеть?"
    with patch("agent.agent.requests.post") as mock_post:
        mock_post.return_value = _ok_response("Нейронная сеть — модель.")
        generate_answer(question, ["Нейронная сеть — математическая модель."])
        sent_json = mock_post.call_args[1]["json"]
    assert question in sent_json["prompt"]


def test_context_is_included_in_prompt():
    """Retrieved RAG chunks must appear in the prompt so the model can use them."""
    chunk = "Уникальный-контекстный-фрагмент-для-проверки"
    with patch("agent.agent.requests.post") as mock_post:
        mock_post.return_value = _ok_response("Ответ на основе контекста.")
        generate_answer("Вопрос?", [chunk])
        sent_json = mock_post.call_args[1]["json"]
    assert chunk in sent_json["prompt"]


def test_empty_context_skips_ollama_and_returns_fallback():
    """Empty context must return the no-info message without making any HTTP call."""
    with patch("agent.agent.requests.post") as mock_post:
        result = generate_answer("Вопрос без контекста", [])
    assert not mock_post.called
    assert "нет информации" in result.lower()
