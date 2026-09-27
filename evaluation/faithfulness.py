from src.config import CHAT_MODEL
from langchain_google_genai import ChatGoogleGenerativeAI

FAITHFULNESS_PROMPT = """You are checking whether an AI's answer is faithful to the given context.

Context:
{context}

Answer to check:
{answer}

If the answer is a refusal (e.g., "I don't know based on the provided documents"),
that is considered fully faithful, since refusing rather than guessing is the correct
behavior when the context doesn't contain the answer. In that case, respond with:
Claims: (refusal, no factual claims to check)
Score: 1

Otherwise, break the answer into individual factual claims. For each claim, check if
it is directly supported by the context above. Then respond in exactly this format:

Claims: <list each claim on its own line, marked SUPPORTED or NOT SUPPORTED>
Score: <a number from 0 to 1, the fraction of claims that were SUPPORTED>
"""

def check_faithfulness(answer, context):
    """Ask Gemini to judge whether the answer's claims are supported by the context."""
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0)
    prompt = FAITHFULNESS_PROMPT.format(context=context, answer=answer)
    response = llm.invoke(prompt)
    return response.content


def extract_score(faithfulness_text):
    """Pull the numeric score out of the judge's response text."""
    for line in faithfulness_text.splitlines():
        if line.strip().lower().startswith("score:"):
            try:
                return float(line.split(":")[1].strip())
            except (ValueError, IndexError):
                return None
    return None