from langchain_text_splitters import RecursiveCharacterTextSplitter
from loaders import load_pdfs, load_website

CHUNK_SIZE = 1000      # target size of each chunk, in characters
CHUNK_OVERLAP = 100 


def split_documents(documents):
    """Cut a list of Documents into smaller chunks. Returns a longer list of Documents."""

    # This splitter tries to cut at paragraph breaks first, then sentences,
    # then words, only falling back to a hard cut if nothing else fits.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(documents)
    return chunks


if __name__ == "__main__":
    pdf_docs = load_pdfs()
    web_docs = load_website()

    all_docs = pdf_docs + web_docs   # combine both lists into one
    chunks = split_documents(all_docs)

    print(f"Documents before chunking: {len(all_docs)}")
    print(f"Chunks after chunking: {len(chunks)}")

    print("\n--- Example chunk ---")
    print(f"Length: {len(chunks[10].page_content)} characters")
    print(f"Metadata: {chunks[10].metadata}")
    print(chunks[10].page_content)