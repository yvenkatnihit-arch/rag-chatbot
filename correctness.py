from config import CHAT_MODEL
from langchain_google_genai import ChatGoogleGenerativeAI

CORRECTNESS_PROMPT = """Compare the generated answer to the reference (known-correct) answer.

Reference answer:
{reference}

Generated answer:
{generated}

Does the generated answer convey the same key facts as the reference answer, even if
worded differently? Respond in exactly this format:

Verdict: <CORRECT or INCORRECT>
Reason: <one short sentence explaining why>
"""


def check_correctness(generated_answer, reference_answer):
    """Ask Gemini to judge whether the generated answer matches the known-correct one."""
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0)
    prompt = CORRECTNESS_PROMPT.format(reference=reference_answer, generated=generated_answer)
    response = llm.invoke(prompt)
    return response.content