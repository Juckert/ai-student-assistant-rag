import os
import faiss
import pickle
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# MODEL
def get_model():
    return SentenceTransformer("intfloat/multilingual-e5-base")


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
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# CHUNKING
def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


# INGEST
def ingest(file_path):
    model = get_model()

    if file_path.endswith(".pdf"):
        text = read_pdf(file_path)
    elif file_path.endswith(".txt"):
        text = read_txt(file_path)
    else:
        raise ValueError("Unsupported file type")

    chunks = chunk_text(text)

    embeddings = model.encode(chunks, convert_to_numpy=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, chunks


# SAVE DB
def save_db(index, chunks):
    os.makedirs("db", exist_ok=True)

    faiss.write_index(index, "db/index.faiss")

    with open("db/docs.pkl", "wb") as f:
        pickle.dump(chunks, f)


# LOAD DB
def load_db():
    if not os.path.exists("db/index.faiss"):
        return None, None

    index = faiss.read_index("db/index.faiss")

    with open("db/docs.pkl", "rb") as f:
        chunks = pickle.load(f)

    return index, chunks