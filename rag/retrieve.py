import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-base")


def search(query, index, chunks, k=1):

    query_vec = model.encode(
        ["query: " + query],
        convert_to_numpy=True
    )

    distances, indices = index.search(query_vec, k)

    results = []
    seen = set()

    for i in indices[0]:
        if i < len(chunks):
            chunk = chunks[i]

            if chunk not in seen:
                results.append(chunk)
                seen.add(chunk)

    return results