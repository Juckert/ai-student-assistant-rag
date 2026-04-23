# ИИ-ассистент студента

Прототип системы вопросов и ответов для студентов программ по искусственному интеллекту.

Проект использует связку `RAG + LLM`:
- база знаний собирается из файлов `PDF`, `TXT` и `CSV`;
- система ищет релевантные фрагменты в базе знаний;
- итоговый ответ формируется локальной моделью через `Ollama`;
- пользователю показываются найденные источники.

## Возможности системы

- обработка вопросов студентов в свободной форме;
- поиск релевантной информации в базе знаний;
- генерация ответа только на основе найденного контекста;
- отображение источников ответа;
- обновление и переиндексация базы знаний через интерфейс.

## Архитектура

Система состоит из трех частей:

1. `RAG`
   Загрузка документов, разбиение на фрагменты, построение эмбеддингов, индексирование и поиск.

2. `Agent`
   Получает вопрос пользователя и найденные RAG-фрагменты, после чего формирует итоговый ответ через `Ollama`.

3. `Web-интерфейс`
   Содержит режим студента для вопросов и режим администратора для загрузки и обновления базы знаний.

## Структура проекта

```text
ai-student-assistant-rag-main/
├─ agent/
│  └─ agent.py
├─ app/
│  └─ app.py
├─ data/
├─ db/
├─ rag/
│  ├─ ingest.py
│  └─ retrieve.py
├─ .gitignore
└─ README.md
```

## Используемый стек

- `Python`
- `Streamlit`
- `Ollama`
- `Transformers`
- `PyTorch`
- `FAISS` или `NumPy`
- `pypdf`

## Поддерживаемые форматы данных

- `.pdf`
- `.txt`
- `.csv`

Для `CSV` используются поля:
- `question_text`
- `answer_text`
- `question_topic`
- `question_year`
- `question_course`

## Запуск проекта

```bash 
docker compose up -d
```

После запуска откройте:

```text
http://localhost:8501
```

## Работа с системой

### Режим `Admin`

Позволяет:
- загружать новые файлы;
- выбирать уже существующие файлы из папки `data`;
- собирать индекс по одному файлу или по всей папке `data`.

После индексации база знаний сохраняется в папке `db/`.

### Режим `Student`

Позволяет:
- ввести вопрос в свободной форме;
- получить итоговый ответ;
- увидеть блок `Sources` с найденными источниками.

## Тесты AI Student Assistant

### Обзор

Тестовый набор покрывает два ключевых слоя приложения:

| Модуль | Файл | Тестов |
|--------|------|--------|
| Агент (Ollama) | `tests/test_agent.py` | 5 |
| RAG (ingest + search) | `tests/test_rag.py` | 7 |

**Внешние зависимости при тестировании:**

| Зависимость | В тестах |
|-------------|----------|
| Ollama | заглушен (`unittest.mock.patch`) |
| torch / transformers | заглушён через `sys.modules` в `conftest.py` |
| FAISS | подменён на `None` → используется `NumpyIndex` |
| Реальная модель эмбеддингов | заменена на `FakeEmbeddingModel` |

Тесты работают без GPU, без интернета, без запущенного Ollama.

---

### Структура

```
tests/
├── conftest.py       # заглушки тяжёлых пакетов, sys.path
├── test_agent.py     # тесты агента
└── test_rag.py       # тесты RAG-пайплайна
pytest.ini            # конфигурация pytest
```

---

#### Тесты

##### `test_message_is_sent_to_ollama`
Проверяет, что при вопросе с непустым контекстом выполняется ровно один `POST`-запрос.

```
generate_answer("Что такое Python?", ["Python — язык программирования."])
→ requests.post.call_count == 1
```

##### `test_response_text_is_returned`
Проверяет, что текст из JSON-ответа Ollama возвращается вызывающему коду без изменений.

```
Ollama вернул: "Python — высокоуровневый язык."
generate_answer(...) вернул: "Python — высокоуровневый язык."
```

##### `test_question_is_included_in_prompt`
Проверяет, что текст вопроса попадает в поле `prompt` JSON-payload, отправляемого в Ollama.

```
question = "Что такое нейронная сеть?"
→ question in sent_json["prompt"]
```

##### `test_context_is_included_in_prompt`
Проверяет, что RAG-фрагменты (чанки) включены в prompt, чтобы модель отвечала на основе контекста.

```
chunk = "Уникальный-контекстный-фрагмент"
→ chunk in sent_json["prompt"]
```

##### `test_empty_context_skips_ollama_and_returns_fallback`
Проверяет, что при пустом контексте HTTP-запрос не отправляется и возвращается fallback-сообщение.

```
generate_answer("Вопрос", [])
→ requests.post не вызван
→ "нет информации" in result.lower()
```

---

### test_rag.py

Тестирует `rag/ingest.py` и `rag/retrieve.py`.

#### FakeEmbeddingModel

Лёгкая детерминированная заглушка реальной модели эмбеддингов.

```python
class FakeEmbeddingModel:
    DIM = 16

    def encode(self, texts, ...):
        seed = sum(ord(c) for c in "|".join(texts)) % (2 ** 32)
        rng = np.random.default_rng(seed=seed)
        vecs = rng.standard_normal((len(texts), DIM))
        return vecs / norm(vecs)  # нормализованные unit-векторы
```

- Один и тот же текст → один и тот же вектор (детерминировано через seed из контрольной суммы символов).
- Разные тексты → разные векторы (с высокой вероятностью).
- Не требует torch, transformers, GPU.

#### Fixture `_patch_embedding_model` (autouse)

Применяется ко **всем** тестам в файле автоматически:

```python
@pytest.fixture(autouse=True)
def _patch_embedding_model():
    fake = FakeEmbeddingModel()
    with patch("rag.ingest.get_model", return_value=fake), \
         patch("rag.retrieve.get_model", return_value=fake), \
         patch("rag.ingest.faiss", None):   # → NumpyIndex вместо FAISS
        yield
```

`faiss` подменяется на `None`, чтобы `ingest()` создавал `NumpyIndex` — чистый NumPy, без бинарников FAISS.

### Запуск

#### Локально

```bash
python -m pytest tests/ -v
```

#### Только агент

```bash
python -m pytest tests/test_agent.py -v
```

#### Только RAG

```bash
python -m pytest tests/test_rag.py -v
```

#### С покрытием

```bash
python -m pytest tests/ -v --cov=agent --cov=rag --cov-report=term-missing
```

---

### Интеграция в Docker

#### При старте контейнера

`entrypoint.sh` запускает тесты до поднятия Streamlit. Вывод виден в `docker compose up`:

```
========================================
  Running test suite
========================================
tests/test_agent.py::test_message_is_sent_to_ollama PASSED
...
12 passed in 0.18s
========================================
  All tests passed — starting assistant
========================================
```

Если тест падает — контейнер останавливается, приложение не стартует.

#### Healthcheck (docker-compose.yml)

```yaml
healthcheck:
  test: ["CMD-SHELL", "python -m pytest /app/tests/ -v --tb=short > /proc/1/fd/1 2>&1"]
  interval: 300s   # каждые 5 минут
  timeout: 60s
  retries: 3
  start_period: 120s
```

Вывод редиректится в stdout контейнера (`/proc/1/fd/1`) и доступен через `docker logs assistant`.
