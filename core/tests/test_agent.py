"""
Tests for the no-info pipeline path and NO_INFO_USER_MESSAGE constant.

The heavy retrieval stack is fully mocked — no LanceDB, no Ollama connection required.
"""
from unittest.mock import MagicMock, patch

import pytest

from rag_system.utils.constants import NO_INFO_USER_MESSAGE
from rag_system.pipelines.retrieval_pipeline import RetrievalPipeline


_CONFIG = {
    "storage": {"db_path": "/tmp/test_lancedb", "text_table_name": "test_table"},
    "retrieval_k": 5,
    "context_window_size": 0,
    "reranker": {"enabled": False},
    "provence": {"enabled": False},
}


@pytest.fixture()
def pipeline():
    mock_client = MagicMock()
    mock_ollama_cfg = {"generation_model": "test-model", "host": "http://localhost:11434"}
    return RetrievalPipeline(_CONFIG.copy(), mock_client, mock_ollama_cfg)


# ── constant ───────────────────────────────────────────────────────────────

def test_no_info_message_is_non_empty_string():
    assert isinstance(NO_INFO_USER_MESSAGE, str) and NO_INFO_USER_MESSAGE


def test_no_info_message_mentions_admin():
    assert "администратор" in NO_INFO_USER_MESSAGE.lower()


# ── pipeline behaviour ─────────────────────────────────────────────────────

def test_pipeline_returns_no_info_when_retriever_finds_nothing(pipeline):
    """Empty retrieval result must produce NO_INFO_USER_MESSAGE, not an LLM call."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []

    with patch.object(pipeline, "_get_dense_retriever", return_value=mock_retriever), \
         patch.object(pipeline, "_get_reranker", return_value=None), \
         patch.object(pipeline, "_get_ai_reranker", return_value=None):
        result = pipeline.run("Вопрос без ответа в базе?")

    assert result["answer"] == NO_INFO_USER_MESSAGE
    assert result["source_documents"] == []


def test_pipeline_no_info_does_not_call_llm(pipeline):
    """LLM synthesis must not be called when there are no retrieved docs."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []

    with patch.object(pipeline, "_get_dense_retriever", return_value=mock_retriever), \
         patch.object(pipeline, "_get_reranker", return_value=None), \
         patch.object(pipeline, "_get_ai_reranker", return_value=None), \
         patch.object(pipeline, "_synthesize_final_answer") as mock_synth:
        pipeline.run("Вопрос без ответа?")

    mock_synth.assert_not_called()
