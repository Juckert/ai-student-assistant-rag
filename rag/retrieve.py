import re

from rag.ingest import get_model

TOKEN_PATTERN = re.compile(r"[A-Za-zА-Яа-яЁё0-9-]+")
STOPWORDS = {
    "а",
    "без",
    "будет",
    "бы",
    "в",
    "во",
    "вообще",
    "вот",
    "вы",
    "где",
    "да",
    "для",
    "до",
    "если",
    "еще",
    "ещё",
    "же",
    "за",
    "и",
    "из",
    "или",
    "к",
    "как",
    "какая",
    "какие",
    "какой",
    "ко",
    "ли",
    "мне",
    "можно",
    "мы",
    "на",
    "надо",
    "не",
    "нет",
    "но",
    "ну",
    "о",
    "об",
    "он",
    "она",
    "они",
    "от",
    "по",
    "под",
    "после",
    "просто",
    "раз",
    "с",
    "со",
    "так",
    "там",
    "то",
    "только",
    "у",
    "уже",
    "что",
    "чтобы",
    "это",
    "этот",
    "этого",
    "эту",
    "я",
}


def tokenize(text):
    return {
        token.lower()
        for token in TOKEN_PATTERN.findall(text)
        if len(token) >= 3 and token.lower() not in STOPWORDS
    }


def lexical_overlap(query, chunk):
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0

    chunk_tokens = tokenize(chunk)
    return len(query_tokens & chunk_tokens)


def search(query, index, chunks, k=1):
    if not query.strip():
        return []

    model = get_model()
    # Fetch a wider pool first, then re-rank it with simple lexical overlap.
    fetch_k = min(len(chunks), max(k * 4, 8))

    query_vec = model.encode(
        ["query: " + query],
        convert_to_numpy=True
    )

    distances, indices = index.search(query_vec, fetch_k)

    candidates = []
    seen_chunks = set()

    for position, chunk_index in enumerate(indices[0]):
        if chunk_index >= len(chunks):
            continue

        chunk = chunks[chunk_index]
        if chunk in seen_chunks:
            continue

        seen_chunks.add(chunk)
        candidates.append(
            {
                "chunk": chunk,
                "distance": float(distances[0][position]),
                # Overlap helps filter semantically close but off-topic chunks.
                "overlap": lexical_overlap(query, chunk),
            }
        )

    if not candidates:
        return []

    candidates.sort(key=lambda item: (-item["overlap"], item["distance"]))

    results = []
    best_distance = candidates[0]["distance"]
    best_overlap = candidates[0]["overlap"]

    for candidate in candidates:
        if len(results) >= k:
            break

        if not results:
            results.append(candidate["chunk"])
            continue

        is_close_enough = candidate["distance"] <= best_distance + 0.08
        has_enough_overlap = (
            candidate["overlap"] > 0 and candidate["overlap"] >= max(1, best_overlap - 1)
        )

        # Keep only candidates that are close to the best hit both semantically and lexically.
        if is_close_enough and has_enough_overlap:
            results.append(candidate["chunk"])

    return results
