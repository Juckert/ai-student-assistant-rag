"""
Tests for rag/ingest.py and rag/retrieve.py.

The heavy transformer embedding model is replaced with FakeEmbeddingModel
(lightweight NumPy stub) — no Ollama, no model download required.

Each test creates its own temporary document and deletes it immediately
after the assertions, even if the test fails (pytest tmp_path handles cleanup).
"""
import os

import numpy as np
import pytest
from unittest.mock import patch

from rag.ingest import ingest, save_db, load_db
from rag.retrieve import search


# ── fake embedding model ───────────────────────────────────────────────────

class FakeEmbeddingModel:
    """
    Deterministic, zero-dependency stub for EmbeddingModel.
    Returns unit vectors derived from the input text so that
    different texts produce different (but reproducible) embeddings.
    """
    DIM = 16

    def encode(self, texts, convert_to_numpy=True, batch_size=16):
        if isinstance(texts, str):
            texts = [texts]
        seed = sum(ord(c) for c in "|".join(texts)) % (2 ** 32)
        rng = np.random.default_rng(seed=seed)
        vecs = rng.standard_normal((len(texts), self.DIM)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.maximum(norms, 1e-9)


# ── autouse fixture: replace get_model in both modules ────────────────────

@pytest.fixture(autouse=True)
def _patch_embedding_model():
    fake = FakeEmbeddingModel()
    # faiss is stubbed to None so NumpyIndex is used — no real FAISS binary needed.
    with patch("rag.ingest.get_model", return_value=fake), \
         patch("rag.retrieve.get_model", return_value=fake), \
         patch("rag.ingest.faiss", None):
        yield


# ── helpers ────────────────────────────────────────────────────────────────

def _assert_deleted(path):
    assert not os.path.exists(str(path)), (
        f"Тестовый документ должен быть удалён после теста: {path}"
    )


# ── document upload tests ──────────────────────────────────────────────────

def test_ingest_txt_document(tmp_path):
    """TXT file is ingested into a valid index with the expected chunks."""
    doc = tmp_path / "knowledge.txt"
    doc.write_text(
        "Python — высокоуровневый язык программирования. " * 15,
        encoding="utf-8",
    )

    index, chunks = ingest(str(doc))

    assert index is not None
    assert len(chunks) >= 1
    assert any("Python" in c for c in chunks)

    doc.unlink()
    _assert_deleted(doc)


def test_ingest_csv_document(tmp_path):
    """CSV file with Q&A rows is ingested; each row becomes one chunk."""
    doc = tmp_path / "qa.csv"
    doc.write_text(
        "question_text,answer_text,question_topic\n"
        "Что такое Python?,Язык программирования.,Программирование\n"
        "Что такое RAG?,Retrieval-Augmented Generation.,ИИ\n",
        encoding="utf-8",
    )

    index, chunks = ingest(str(doc))

    assert index is not None
    assert len(chunks) == 2
    assert any("Python" in c for c in chunks)
    assert any("RAG" in c for c in chunks)

    doc.unlink()
    _assert_deleted(doc)


def test_ingest_multiple_files(tmp_path):
    """Ingesting a list of files merges all chunks into one index."""
    doc1 = tmp_path / "a.txt"
    doc2 = tmp_path / "b.txt"
    doc1.write_text("Тема один: алгоритмы и структуры данных. " * 10, encoding="utf-8")
    doc2.write_text("Тема два: машинное обучение и нейросети. " * 10, encoding="utf-8")

    index, chunks = ingest([str(doc1), str(doc2)])

    assert index is not None
    assert len(chunks) >= 2
    full_text = " ".join(chunks)
    assert "алгоритмы" in full_text
    assert "машинное" in full_text

    doc1.unlink()
    doc2.unlink()
    _assert_deleted(doc1)
    _assert_deleted(doc2)


# ── save / load DB ─────────────────────────────────────────────────────────

def test_save_and_load_db_roundtrip(tmp_path):
    """Chunks persisted with save_db are recovered identically by load_db."""
    doc = tmp_path / "persist.txt"
    doc.write_text("Тест сохранения и загрузки базы знаний.", encoding="utf-8")

    with patch("rag.ingest.DB_DIR", str(tmp_path / "db")):
        index, chunks = ingest(str(doc))
        save_db(index, chunks)
        loaded_index, loaded_chunks = load_db()

    assert loaded_chunks == chunks
    assert loaded_index is not None

    doc.unlink()
    _assert_deleted(doc)


# ── search tests ───────────────────────────────────────────────────────────

def test_search_returns_list_of_strings(tmp_path):
    """search() must return a list where every element is a string."""
    doc = tmp_path / "search_test.txt"
    doc.write_text(
        "Искусственный интеллект — область компьютерных наук. " * 10,
        encoding="utf-8",
    )

    index, chunks = ingest(str(doc))
    results = search("Что такое искусственный интеллект?", index, chunks, k=2)

    assert isinstance(results, list)
    assert all(isinstance(r, str) for r in results)

    doc.unlink()
    _assert_deleted(doc)


def test_search_respects_k_limit(tmp_path):
    """search() must return at most k results."""
    doc = tmp_path / "k_limit.txt"
    doc.write_text(
        ("Фрагмент о базах данных. " * 30 + "\n") * 5,
        encoding="utf-8",
    )

    index, chunks = ingest(str(doc))
    results = search("база данных", index, chunks, k=1)

    assert len(results) <= 1

    doc.unlink()
    _assert_deleted(doc)


def test_search_empty_query_returns_empty(tmp_path):
    """A blank query must return an empty list without raising."""
    doc = tmp_path / "empty_query.txt"
    doc.write_text("Любой текст для индексации.", encoding="utf-8")

    index, chunks = ingest(str(doc))
    results = search("   ", index, chunks, k=2)

    assert results == []

    doc.unlink()
    _assert_deleted(doc)
