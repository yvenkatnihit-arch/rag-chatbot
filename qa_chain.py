from config import CHAT_MODEL
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from vector_store import load_vector_store
from citations import get_unique_citations
from dedup import deduplicate_chunks
from hybrid_search import get_bm25_index, bm25_search, reciprocal_rank_fusion
from reranker import rerank_chunks

USE_HYBRID_SEARCH = True # flip to False to compare against semantic-only
USE_RERANKER = False # flip to False to compare against no reranking

PROMPT_TEMPLATE = """You are a helpful assistant answering questions using ONLY the context below.

Rules:
- Only use information found in the context. Do not use any outside knowledge.
- If the context does not contain the answer, say exactly: "I don't know based on the provided documents."
- Do not make anything up.
- Each piece of context is labeled with its source. When you state a fact, make sure it actually comes from the source you cite for it.
- Include all relevant details from the context, not just the first or most obvious ones. If the context lists multiple items, list all of them.

Context:
{context}

Question: {question}

Answer:"""

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


def retrieve_chunks(question, k=4):
    """Retrieve chunks for a question. Uses hybrid search if enabled, then
    optionally reranks with a cross-encoder for more precise final ordering."""

    # When reranking, fetch a wider candidate pool so the reranker has choices
    candidate_k = 8 if USE_RERANKER else k + 2

    vector_store = load_vector_store()
    semantic_results = vector_store.similarity_search(question, k=candidate_k)

    if not USE_HYBRID_SEARCH:
        chunks = deduplicate_chunks(semantic_results)
    else:
        bm25, texts, metadatas = get_bm25_index()
        keyword_results = bm25_search(bm25, texts, metadatas, question, k=candidate_k)

        merged = reciprocal_rank_fusion(semantic_results, keyword_results, k=candidate_k)

        chunks = []
        for source_type, item in merged:
            if source_type == "semantic":
                chunks.append(item)
            else:
                text, metadata = item
                chunks.append(Document(page_content=text, metadata=metadata))

        chunks = deduplicate_chunks(chunks)

    # Rerank the candidate pool down to the final k
    if USE_RERANKER:
        chunks = rerank_chunks(question, chunks, top_k=k)
    else:
        chunks = chunks[:k]

    return chunks


def answer_question(question, history=None, k=4):
    """Retrieve relevant chunks, build a prompt, and get a grounded answer from Gemini."""

    if history is None:
        history = []

    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0)

    standalone_question = rewrite_as_standalone(question, history, llm)

    chunks = retrieve_chunks(standalone_question, k=k)

    context = format_chunks_as_context(chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=standalone_question)

    response = llm.invoke(prompt)

    return response.content, chunks


def answer_question_streaming(question, history=None, k=4):
    """Same as answer_question, but yields the answer piece by piece as it's generated."""

    if history is None:
        history = []

    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0)

    standalone_question = rewrite_as_standalone(question, history, llm)

    chunks = retrieve_chunks(standalone_question, k=k)

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
        "What is the power output of the Bugatti Veyron?",
        "Who is associated with roll number 24MC201A96?",
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