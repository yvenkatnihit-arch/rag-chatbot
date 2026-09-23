def deduplicate_chunks(chunks, similarity_threshold=0.9):
    """Remove chunks that are near-duplicates of chunks already kept.
    Keeps the first occurrence, drops later ones that are too similar in raw text."""

    def text_overlap_ratio(text_a, text_b):
        # A simple, fast measure of text similarity: how much of the shorter text's
        # words appear in the longer text. Not as precise as embeddings, but fast
        # and good enough for catching near-duplicates from chunk overlap.
        words_a = set(text_a.split())
        words_b = set(text_b.split())
        if not words_a or not words_b:
            return 0
        overlap = len(words_a & words_b)          # words in both
        smaller = min(len(words_a), len(words_b))
        return overlap / smaller

    kept_chunks = []
    for chunk in chunks:
        is_duplicate = False
        for kept in kept_chunks:
            if text_overlap_ratio(chunk.page_content, kept.page_content) >= similarity_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            kept_chunks.append(chunk)

    return kept_chunks