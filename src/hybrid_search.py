import re
from rank_bm25 import BM25Okapi
from .vector_store import load_vector_store


def tokenize(text):
    """Lowercase and split into words, stripping punctuation so tokens match
    cleanly (e.g. '24MC201A96?' and '24mc201a96' become the same token)."""
    text = text.lower()
    words = re.findall(r"[a-z0-9]+", text)
    return words


def build_bm25_index(vector_store, b=0.3):
    """Pull every chunk's text out of Chroma and build a BM25 keyword index over it.
    b controls document-length normalization (0.75 = default, lower = less penalty
    for long chunks). Lowered here because our corpus has uneven chunk lengths."""

    raw_data = vector_store.get(include=["documents", "metadatas"])

    texts = raw_data["documents"]
    metadatas = raw_data["metadatas"]

    tokenized_texts = [tokenize(text) for text in texts]
    bm25 = BM25Okapi(tokenized_texts, b=b)

    return bm25, texts, metadatas


def bm25_search(bm25, texts, metadatas, question, k=4):
    """Run a keyword search and return the top k chunks as (text, metadata) pairs."""
    tokenized_question = tokenize(question)
    scores = bm25.get_scores(tokenized_question)

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    results = [(texts[i], metadatas[i]) for i in ranked_indices]
    return results


def reciprocal_rank_fusion(semantic_results, keyword_results, k=4,
                           rrf_constant=60, semantic_weight=0.7, keyword_weight=0.3):
    """Merge two ranked lists into one, weighting semantic search more heavily
    since it performs better on natural-language questions. Keyword search acts
    as a safety net for exact terms rather than an equal competitor."""

    scores = {}
    all_chunks = {}

    for rank, chunk in enumerate(semantic_results):
        key = chunk.page_content
        scores[key] = scores.get(key, 0) + semantic_weight / (rrf_constant + rank)
        all_chunks[key] = ("semantic", chunk)

    for rank, (text, metadata) in enumerate(keyword_results):
        key = text
        scores[key] = scores.get(key, 0) + keyword_weight / (rrf_constant + rank)
        all_chunks[key] = ("keyword", (text, metadata))

    ranked_keys = sorted(scores.keys(), key=lambda key: scores[key], reverse=True)
    return [all_chunks[key] for key in ranked_keys[:k]]


# Build the BM25 index ONCE when this module is first imported, rather than
# rebuilding it on every question. Important for performance at larger scale.
_vector_store = load_vector_store()
_bm25, _texts, _metadatas = build_bm25_index(_vector_store)


def get_bm25_index():
    """Return the pre-built BM25 index, built once on first import."""
    return _bm25, _texts, _metadatas


# The BM25 index is built lazily, on first use, rather than at import time.
# This avoids crashing if chroma_db doesn't exist yet, and avoids the cost
# of indexing 787 chunks when hybrid search isn't actually being used.
_bm25_cache = None


def get_bm25_index():
    """Return the BM25 index, building it on first call and reusing it afterward."""
    global _bm25_cache

    if _bm25_cache is None:
        vector_store = load_vector_store()
        _bm25_cache = build_bm25_index(vector_store)

    return _bm25_cache