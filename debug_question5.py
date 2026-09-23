from vector_store import load_vector_store

vector_store = load_vector_store()

question = "Which product category generates the most revenue?"
chunks = vector_store.similarity_search(question, k=4)

for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i + 1} ---")
    print(f"Source: {chunk.metadata.get('source')}, page: {chunk.metadata.get('page')}")
    print(chunk.page_content)
    print()