import os                          # lets us read environment variables
from dotenv import load_dotenv     # reads the .env file

load_dotenv()                      # copies values from .env into the environment

key = os.getenv("GOOGLE_API_KEY")  # look up the key by its name

if key:
    # Print the length only, so the secret itself never appears on screen
    print(f"API key loaded (length {len(key)})")
else:
    print("No API key found. Check your .env file.")

# If any of these imports fail, that library didn't install properly
import langchain, langchain_google_genai, langchain_chroma, pypdf, bs4
print("All libraries imported successfully")