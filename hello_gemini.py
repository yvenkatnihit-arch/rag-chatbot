from config import CHAT_MODEL  # importing config also loads .env
from langchain_google_genai import ChatGoogleGenerativeAI  # LangChain's Gemini connector

# Create the model object. It finds GOOGLE_API_KEY in the environment by itself.
llm = ChatGoogleGenerativeAI(
    model=CHAT_MODEL,   # which Gemini model to use
    temperature=0.3,      # 0 = focused and predictable answers
)

# invoke() sends the prompt to Google and waits for the full reply
response = llm.invoke("give me few ideas about buiilding a good ai project")

print(response.content)           # the text of the answer
print(response.usage_metadata)    # how many tokens this call used