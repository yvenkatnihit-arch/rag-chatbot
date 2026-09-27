from sentence_transformers import CrossEncoder

# A small, fast, widely-used reranking model. Downloads automatically on first use
# and then runs locally, no API calls, no cost per query.
RERANKER_MODEL = "BAAI/bge-reranker-base"

# Load the model once when this module is first imported, not per query
_model = CrossEncoder(RERANKER_MODEL)


def rerank_chunks(question, chunks, top_k=4):
    """Score each chunk against the question using a cross-encoder, then return
    the top_k highest-scoring chunks in score order.

    Unlike embedding similarity, the cross-encoder reads the question and chunk
    together, so it can judge whether the chunk actually answers the question."""

    if not chunks:
        return []

    # Build (question, chunk_text) pairs, the format the cross-encoder expects
    pairs = [(question, chunk.page_content) for chunk in chunks]

    # One score per pair. Higher = more relevant.
    scores = _model.predict(pairs)

    # Pair each chunk with its score, sort by score descending
    scored_chunks = list(zip(chunks, scores))
    scored_chunks.sort(key=lambda pair: pair[1], reverse=True)

    # Return just the chunks, dropping the scores
    return [chunk for chunk, score in scored_chunks[:top_k]]