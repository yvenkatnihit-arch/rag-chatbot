def format_citation(chunk):
    """Turn one chunk's metadata into a readable citation string."""
    source = chunk.metadata.get("source", "unknown source")

    if source.startswith("http"):
        # It's a website. Metadata has no page number, so just show the URL.
        return source
    else:
        # It's a PDF. metadata['page'] is 0-based, so add 1 for the human-readable page number.
        page = chunk.metadata.get("page")
        if page is not None:
            page_number = page + 1
            return f"{source} (page {page_number})"
        else:
            return source


def get_unique_citations(chunks):
    """Build a citation list from chunks, without repeating the same source twice."""
    citations = []
    seen = set()   # keeps track of citations we've already added

    for chunk in chunks:
        citation = format_citation(chunk)
        if citation not in seen:
            citations.append(citation)
            seen.add(citation)

    return citations

def is_refusal(answer):
    """Check whether the answer is a refusal, so we can skip showing sources."""
    return "i don't know based on the provided documents" in answer.lower()