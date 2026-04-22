import os

import requests
from requests import RequestException

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")


def generate_answer(question, context_chunks):
    """
    Build a grounded answer from the retrieved RAG context using Ollama.
    """
    context = "\n\n".join(context_chunks)

    if not context.strip():
        return "В базе знаний нет информации."

    prompt = f"""
Ты — AI-ассистент студента.

Твоя задача:
- Отвечать только на основе контекста
- Если ответа нет в контексте — скажи: "В базе знаний нет информации."
- Не придумывай ничего от себя
- Если в контексте есть прямой ответ, передай его максимально конкретно и кратко
- Не добавляй предположения, общие советы или фразы вроде "обратитесь к консультанту", если этого нет в контексте
- Если вопрос спрашивает про другой год, срок или условие, а в контексте есть только похожая информация, честно скажи, что точной информации нет, и кратко укажи, что именно есть в базе

Контекст:
{context}

Вопрос:
{question}

Ответ:
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=120,
        )
        response.raise_for_status()
    except RequestException as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_URL}. "
            f"Make sure the server is running and model '{MODEL}' is installed."
        ) from exc

    payload = response.json()
    answer = (payload.get("response") or "").strip()
    return answer or "No answer was returned by the model."
