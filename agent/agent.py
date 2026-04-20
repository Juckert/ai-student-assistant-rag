import requests

# локальный Ollama сервер
OLLAMA_URL = "http://localhost:11434/api/generate"

MODEL = "qwen2.5:7b"


def generate_answer(question, context_chunks):
    """
    Агент получает:
    - вопрос пользователя
    - найденные RAG chunks

    И возвращает финальный ответ.
    """

    # собираем контекст
    context = "\n\n".join(context_chunks)

    # RAG prompt
    prompt = f"""
Ты — AI-ассистент студента.

Твоя задача:
- Отвечать ТОЛЬКО на основе контекста
- Если ответа нет в контексте — скажи: "В базе знаний нет информации"
- Не придумывай ничего от себя

Контекст:
{context}

Вопрос:
{question}

Ответ:
"""

    # запрос в Ollama
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        }
    )

    response.raise_for_status()

    return response.json()["response"]