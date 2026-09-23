from config import CHAT_MODEL
from langchain_google_genai import ChatGoogleGenerativeAI
from vector_store import load_vector_store
from citations import get_unique_citations
from dedup import deduplicate_chunks

# This is the instruction template for the final answer. {context} and {question} get filled in later.
PROMPT_TEMPLATE = """You are a helpful assistant answering questions using ONLY the context below.

Rules:
- Only use information found in the context. Do not use any outside knowledge.
- If the context does not contain the answer, say exactly: "I don't know based on the provided documents."
- Do not make anything up.
- Each piece of context is labeled with its source. When you state a fact, make sure it actually comes from the source you cite for it.
Context:
{context}

Question: {question}

Answer:"""

# This is the instruction template for rewriting a follow-up into a standalone question.
REWRITE_TEMPLATE = """Given the conversation history and a new question, rewrite the new
question as a standalone question that includes any necessary context from the history.
If the new question is already standalone, return it unchanged.

Conversation history:
{history}

New question: {question}

Standalone question:"""


def format_chunks_as_context(chunks):
    """Turn a list of retrieved chunks into one text block for the prompt,
    labeling each piece with its source so the model can cite correctly."""
    pieces = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown source")
        page = chunk.metadata.get("page")
        if page is not None:
            label = f"[Source: {source}, page {page + 1}]"
        else:
            label = f"[Source: {source}]"
        pieces.append(f"{label}\n{chunk.page_content}")
    context = "\n\n---\n\n".join(pieces)
    return context


def format_history(history):
    """Turn a list of (question, answer) pairs into readable text for the prompt."""
    if not history:
        return "No previous conversation."

    lines = []
    for past_question, past_answer in history:
        lines.append(f"Q: {past_question}")
        lines.append(f"A: {past_answer}")
    return "\n".join(lines)


def rewrite_as_standalone(question, history, llm):
    """Use Gemini to rewrite a follow-up question into a standalone one, using history."""
    if not history:
        return question

    history_text = format_history(history)
    prompt = REWRITE_TEMPLATE.format(history=history_text, question=question)
    response = llm.invoke(prompt)
    return response.content.strip()


def answer_question(question, history=None, k=4):
    """Retrieve relevant chunks, build a prompt, and get a grounded answer from Gemini.
    history is a list of (question, answer) tuples from earlier in the conversation."""

    if history is None:
        history = []

    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0)

    standalone_question = rewrite_as_standalone(question, history, llm)

    vector_store = load_vector_store()
    chunks = vector_store.similarity_search(standalone_question, k=k + 2)  # fetch a couple extra
    chunks = deduplicate_chunks(chunks)   # remove near-duplicates
    chunks = chunks[:k]                   # trim back down to kchunks = vector_store.similarity_search(standalone_question, k=k)

    context = format_chunks_as_context(chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=standalone_question)

    response = llm.invoke(prompt)

    return response.content, chunks


def answer_question_streaming(question, history=None, k=4):
    """Same as answer_question, but yields the answer piece by piece as it's generated.
    Used by the Streamlit app."""

    if history is None:
        history = []

    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0)

    standalone_question = rewrite_as_standalone(question, history, llm)

    vector_store = load_vector_store()
    chunks = vector_store.similarity_search(standalone_question, k=k + 2)  # fetch a couple extra
    chunks = deduplicate_chunks(chunks)   # remove near-duplicates
    chunks = chunks[:k]                   # trim back down to kchunks = vector_store.similarity_search(standalone_question, k=k)

    context = format_chunks_as_context(chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=standalone_question)

    def token_generator():
        for piece in llm.stream(prompt):
            yield piece.content

    return token_generator(), chunks


if __name__ == "__main__":
    history = []

    questions = [
        "How many countries does Nike sell its products in?",
        "What percentage of Nike's revenue has come from outside the U.S. since 2005?",
        "What brands are part of Nike's portfolio besides Nike and Jordan?",
    ]

    for question in questions:
        answer, chunks = answer_question(question, history=history)
        citations = get_unique_citations(chunks)

        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print("Sources:")
        for citation in citations:
            print(f"- {citation}")
        print()

        history.append((question, answer))