"""HTTP client for the backend API (port 8000)."""

import json
import os
from typing import Optional

import requests

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
KB_INDEX_NAME = "knowledge_base"

_KB_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "storage", "kb_config.json")


# --- KB config persistence ---

def _load_kb_config() -> dict:
    if os.path.exists(_KB_CONFIG_PATH):
        try:
            with open(_KB_CONFIG_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_kb_config(cfg: dict) -> None:
    os.makedirs(os.path.dirname(_KB_CONFIG_PATH), exist_ok=True)
    with open(_KB_CONFIG_PATH, "w") as f:
        json.dump(cfg, f)


# --- Health ---

def is_backend_available() -> bool:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# --- KB Index management ---

def get_kb_index_id() -> Optional[str]:
    """Return cached KB index ID, or find it from the backend by name."""
    cfg = _load_kb_config()
    if cfg.get("index_id"):
        return cfg["index_id"]
    try:
        r = requests.get(f"{BACKEND_URL}/indexes", timeout=5)
        if r.status_code == 200:
            for idx in r.json().get("indexes", []):
                if idx.get("name") == KB_INDEX_NAME:
                    _save_kb_config({"index_id": idx["id"]})
                    return idx["id"]
    except Exception:
        pass
    return None


def create_kb_index() -> Optional[str]:
    """Create the shared knowledge-base index. Returns its ID."""
    try:
        r = requests.post(
            f"{BACKEND_URL}/indexes",
            json={"name": KB_INDEX_NAME, "description": "Общая база знаний"},
            timeout=10,
        )
        if r.status_code == 201:
            idx_id = r.json()["index_id"]
            _save_kb_config({"index_id": idx_id})
            return idx_id
    except Exception:
        pass
    return None


def get_or_create_kb_index() -> Optional[str]:
    return get_kb_index_id() or create_kb_index()


def upload_files_to_index(index_id: str, files: list[tuple[str, bytes]]) -> list[str]:
    """Upload files to an index. Each element is (filename, bytes). Returns uploaded filenames."""
    uploaded = []
    for filename, content in files:
        try:
            r = requests.post(
                f"{BACKEND_URL}/indexes/{index_id}/upload",
                files={"files": (filename, content)},
                timeout=60,
            )
            if r.status_code == 200:
                uploaded.append(filename)
        except Exception:
            pass
    return uploaded


def build_kb_index(index_id: str) -> tuple[bool, str]:
    """Trigger RAG indexing for the given index. Returns (success, message)."""
    try:
        r = requests.post(
            f"{BACKEND_URL}/indexes/{index_id}/build",
            json={},
            timeout=600,
        )
        if r.status_code == 200:
            data = r.json()
            msg = data.get("message") or data.get("response", {}).get("message", "Индексирование завершено.")
            return True, msg
        return False, r.text
    except requests.exceptions.Timeout:
        return False, "Время ожидания истекло. Индексирование может ещё выполняться на сервере."
    except requests.exceptions.ConnectionError:
        return False, "Не удалось подключиться к backend. Убедитесь, что сервер запущен."
    except Exception as exc:
        return False, str(exc)


# --- Session management ---

def create_backend_session(title: str = "Новый диалог") -> Optional[str]:
    """Create a session in the backend. Returns its session_id (UUID)."""
    try:
        r = requests.post(
            f"{BACKEND_URL}/sessions",
            json={"title": title},
            timeout=10,
        )
        if r.status_code == 201:
            return r.json()["session"]["id"]
    except Exception:
        pass
    return None


def link_index_to_session(session_id: str, index_id: str) -> bool:
    try:
        r = requests.post(
            f"{BACKEND_URL}/sessions/{session_id}/indexes/{index_id}",
            timeout=10,
        )
        return r.status_code == 200
    except Exception:
        return False


# --- Chat ---

def rag_chat(session_id: str, message: str) -> tuple[str, list]:
    """Send a message through the backend (smart RAG routing). Returns (answer, source_documents)."""
    try:
        r = requests.post(
            f"{BACKEND_URL}/sessions/{session_id}/messages",
            json={"message": message},
            timeout=180,
        )
        if r.status_code == 200:
            data = r.json()
            return data.get("response", ""), data.get("source_documents", [])
        return f"Ошибка сервера ({r.status_code}): {r.text}", []
    except requests.exceptions.Timeout:
        return "Время ожидания ответа истекло. Попробуйте повторить запрос.", []
    except requests.exceptions.ConnectionError:
        return "Не удалось подключиться к backend. Убедитесь, что сервер запущен.", []
    except Exception as exc:
        return f"Ошибка: {exc}", []
