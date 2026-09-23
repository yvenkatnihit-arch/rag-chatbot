from config import DATA_DIR, WEBSITE_URL  # first, so USER_AGENT is set before LangChain loads
from pathlib import Path  # handles file paths cleanly on Windows
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader  # the two loaders

def load_pdfs(folder=DATA_DIR):
    """Read every PDF in the folder. Returns a list of Documents, one per page."""
    docs = []
    for pdf_path in sorted(Path(folder).glob("*.pdf")):    # find all .pdf files
        pages = PyPDFLoader(str(pdf_path)).load()          # one Document per PDF page
        docs.extend(pages)                                 # add them to our big list
    return docs


def load_website(url=WEBSITE_URL):
    """Download one web page and extract its text as a Document."""
    return WebBaseLoader(url).load()


# This block only runs when you execute this file directly,
# not when another file imports from it.
if __name__ == "__main__":
    pdf_docs = load_pdfs()
    web_docs = load_website()

    print(f"PDF pages loaded: {len(pdf_docs)}")
    print(f"Website documents loaded: {len(web_docs)}")

    print("\n--- First PDF page ---")
    print(pdf_docs[0].metadata)                 # the label on the back of the card
    print(pdf_docs[0].page_content[:300])       # first 300 characters of text

    print("\n--- Website ---")
    print(web_docs[0].metadata)
    print(web_docs[0].page_content[:300])