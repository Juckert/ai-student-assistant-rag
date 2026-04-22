import csv
import os
import pickle
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from pypdf import PdfReader
from transformers import AutoModel, AutoTokenizer

try:
    import faiss
except ImportError:
    faiss = None

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_DIR = os.path.join(BASE_DIR, "db")
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "intfloat/multilingual-e5-base",
)
FALLBACK_EMBEDDING_MODEL_NAME = os.getenv(
    "FALLBACK_EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2",
)


class NumpyIndex:
    def __init__(self, embeddings):
        self.embeddings = np.asarray(embeddings, dtype=np.float32)

    def search(self, query_vec, k):
        query_vec = np.asarray(query_vec, dtype=np.float32)
        diff = self.embeddings[None, :, :] - query_vec[:, None, :]
        distances = np.sum(diff * diff, axis=2)
        indices = np.argsort(distances, axis=1)[:, :k]
        sorted_distances = np.take_along_axis(distances, indices, axis=1)
        return sorted_distances, indices


class EmbeddingModel:
    def __init__(self, model_name):
        local_model_path = resolve_local_model_path(model_name)

        if local_model_path is not None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    local_model_path,
                    local_files_only=True,
                )
                self.model = AutoModel.from_pretrained(
                    local_model_path,
                    local_files_only=True,
                )
            except Exception:
                local_model_path = None

        if local_model_path is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)

        self.model.eval()

    def encode(self, texts, convert_to_numpy=True, batch_size=16):
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []

        with torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                outputs = self.model(**inputs)

                token_embeddings = outputs.last_hidden_state
                attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
                pooled = (token_embeddings * attention_mask).sum(dim=1)
                pooled = pooled / torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                pooled = F.normalize(pooled, p=2, dim=1)
                embeddings.append(pooled.cpu())

        all_embeddings = torch.cat(embeddings, dim=0)
        return all_embeddings.numpy() if convert_to_numpy else all_embeddings


def resolve_local_model_path(model_name):
    cache_root = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    repo_dir = os.path.join(cache_root, f"models--{model_name.replace('/', '--')}")
    ref_path = os.path.join(repo_dir, "refs", "main")

    if not os.path.exists(ref_path):
        return None

    with open(ref_path, "r", encoding="utf-8") as ref_file:
        snapshot_ref = ref_file.read().strip()

    if not snapshot_ref:
        return None

    snapshot_path = os.path.join(repo_dir, "snapshots", snapshot_ref)
    return snapshot_path if os.path.exists(snapshot_path) else None


def build_embedding_model(model_name):
    return EmbeddingModel(model_name)


# MODEL
@lru_cache(maxsize=1)
def get_model():
    try:
        return build_embedding_model(EMBEDDING_MODEL_NAME)
    except Exception as primary_exc:
        if FALLBACK_EMBEDDING_MODEL_NAME and FALLBACK_EMBEDDING_MODEL_NAME != EMBEDDING_MODEL_NAME:
            try:
                return build_embedding_model(FALLBACK_EMBEDDING_MODEL_NAME)
            except Exception as fallback_exc:
                raise RuntimeError(
                    "Could not load the embedding model. "
                    f"Primary: '{EMBEDDING_MODEL_NAME}' ({primary_exc}). "
                    f"Fallback: '{FALLBACK_EMBEDDING_MODEL_NAME}' ({fallback_exc})."
                ) from fallback_exc

        raise RuntimeError(
            "Could not load the embedding model. "
            f"Primary: '{EMBEDDING_MODEL_NAME}' ({primary_exc})."
        ) from primary_exc


# LOAD TEXT
def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8-sig") as f:
        return f.read()


def read_csv(file_path):
    chunks = []

    with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            question = (row.get("question_text") or "").strip()
            answer = (row.get("answer_text") or "").strip()

            if not question or not answer:
                continue

            topic = (row.get("question_topic") or "").strip()
            year = (row.get("question_year") or "").strip()
            course = (row.get("question_course") or "").strip()

            parts = [
                f"Вопрос: {question}",
                f"Ответ: {answer}",
            ]

            if topic:
                parts.append(f"Тема: {topic}")
            if year:
                parts.append(f"Год: {year}")
            if course:
                parts.append(f"Курс: {course}")

            chunks.append("\n".join(parts))

    return chunks


# CHUNKING
def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


def read_supported_file(file_path):
    lower_path = file_path.lower()

    if lower_path.endswith(".pdf"):
        return chunk_text(read_pdf(file_path))
    if lower_path.endswith(".txt"):
        return chunk_text(read_txt(file_path))
    if lower_path.endswith(".csv"):
        return read_csv(file_path)

    raise ValueError(f"Unsupported file type: {file_path}")


# INGEST
def ingest(file_paths):
    model = get_model()
    all_chunks = []

    if isinstance(file_paths, str):
        file_paths = [file_paths]

    for file_path in file_paths:
        all_chunks.extend(read_supported_file(file_path))

    if not all_chunks:
        raise ValueError("No supported content found for indexing")

    embeddings = model.encode(
        [f"passage: {chunk}" for chunk in all_chunks],
        convert_to_numpy=True
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if faiss is not None:
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
    else:
        index = NumpyIndex(embeddings)

    return index, all_chunks


# SAVE DB
def save_db(index, chunks):
    os.makedirs(DB_DIR, exist_ok=True)
    faiss_path = os.path.join(DB_DIR, "index.faiss")
    embeddings_path = os.path.join(DB_DIR, "embeddings.npy")
    docs_path = os.path.join(DB_DIR, "docs.pkl")

    if faiss is not None and not isinstance(index, NumpyIndex):
        faiss.write_index(index, faiss_path)
        if os.path.exists(embeddings_path):
            os.remove(embeddings_path)
    else:
        np.save(embeddings_path, index.embeddings)
        if os.path.exists(faiss_path):
            os.remove(faiss_path)

    with open(docs_path, "wb") as f:
        pickle.dump(chunks, f)


# LOAD DB
def load_db():
    faiss_path = os.path.join(DB_DIR, "index.faiss")
    embeddings_path = os.path.join(DB_DIR, "embeddings.npy")
    docs_path = os.path.join(DB_DIR, "docs.pkl")
    has_faiss_index = os.path.exists(faiss_path)
    has_numpy_index = os.path.exists(embeddings_path)

    if not has_faiss_index and not has_numpy_index:
        return None, None

    if has_faiss_index and faiss is not None:
        index = faiss.read_index(faiss_path)
    elif has_numpy_index:
        embeddings = np.load(embeddings_path)
        index = NumpyIndex(embeddings)
    else:
        return None, None

    with open(docs_path, "rb") as f:
        chunks = pickle.load(f)

    return index, chunks
