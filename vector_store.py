from config import EMBEDDING_MODEL, CHROMA_DIR, COLLECTION_NAME
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from chunker import split_documents
from loaders import load_pdfs, load_website


def build_vector_store():
    """Embed all chunks and save them to Chroma on disk. Run this once to build the database."""

    # Load and chunk everything, same as before
    pdf_docs = load_pdfs()
    web_docs = load_website()
    all_docs = pdf_docs + web_docs
    chunks = split_documents(all_docs)

    print(f"Embedding and storing {len(chunks)} chunks. This calls the API once per chunk, so it will take a few minutes.")

    # This object knows how to turn text into vectors
    embedder = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    # Chroma.from_documents does three things: embeds every chunk, stores the
    # vectors and text, and saves everything to CHROMA_DIR on disk
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )

    print("Done. Chroma database saved to disk.")
    return vector_store


def load_vector_store():
    """Load the already-built Chroma database from disk, without re-embedding anything."""
    embedder = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedder,
        collection_name=COLLECTION_NAME,
    )
    return vector_store


if __name__ == "__main__":
    vector_store = load_vector_store()   # loads the existing database, no re-embedding

    # Now test a search
    question = "What risks does Nike mention related to its supply chain?"

    # similarity_search embeds the question, then returns the k closest chunks
    results = vector_store.similarity_search(question, k=4)

    print(f"\nQuestion: {question}")
    print(f"Top {len(results)} matching chunks:\n")

    for i, chunk in enumerate(results):
        print(f"--- Match {i + 1} ---")
        print(f"Source: {chunk.metadata.get('source')}, page: {chunk.metadata.get('page')}")
        print(chunk.page_content[:250])
        print()