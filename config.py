from dotenv import load_dotenv  # reads the .env file

load_dotenv()  # runs once, so any file that imports config gets the API key loaded

# Change this one line if Google renames or retires the model
CHAT_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/gemini-embedding-001"
CHROMA_DIR = "chroma_db"          # folder where Chroma saves its data
COLLECTION_NAME = "nike_docs"     # name for this group of stored chunks
import os

os.environ.setdefault("USER_AGENT", "rag-chatbot-learning/0.1")

DATA_DIR = "data"
WEBSITE_URL = "https://en.wikipedia.org/wiki/Nike,_Inc."
