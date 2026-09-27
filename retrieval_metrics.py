from qa_chain import retrieve_chunks
from metrics_data import METRICS_TEST_SET


def is_chunk_relevant(chunk, relevant_sources):
    """Check if a retrieved chunk's source matches one of the expected relevant sources."""
    source = chunk.metadata.get("source", "")
    for expected in relevant_sources:
        if expected in source:
            return True
    return False


def precision_at_k(chunks, relevant_sources):
    """What fraction of retrieved chunks were actually relevant?"""
    if not chunks:
        return 0
    if not relevant_sources:
        return 0
    relevant_count = 0
    for chunk in chunks:
        if is_chunk_relevant(chunk, relevant_sources):
            relevant_count += 1
    return relevant_count / len(chunks)


if __name__ == "__main__":
    total_precision = 0

    for item in METRICS_TEST_SET:
        question = item["question"]
        relevant_sources = item["relevant_sources"]

        chunks = retrieve_chunks(question, k=4)   # now uses the real pipeline
        precision = precision_at_k(chunks, relevant_sources)
        total_precision += precision

        print(f"Q: {question}")
        print(f"Expected source(s): {relevant_sources if relevant_sources else 'NONE (trap question)'}")
        print(f"Precision@4: {precision:.2f}")
        for chunk in chunks:
            mark = "correct" if is_chunk_relevant(chunk, relevant_sources) else "wrong"
            print(f"  - {chunk.metadata.get('source')} ({mark})")
        print()

    average_precision = total_precision / len(METRICS_TEST_SET)
    print(f"Average Precision@4 across all test questions: {average_precision:.2f}")