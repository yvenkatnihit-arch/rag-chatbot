from src.qa_chain import answer_question
from src.citations import get_unique_citations, is_refusal

history = []

print("Ask questions about your documents. Type 'quit' to exit.\n")

while True:
    question = input("You: ")

    if question.lower() in ("quit", "exit", "q"):
        break

    answer, chunks = answer_question(question, history=history)

    print(f"\nAnswer: {answer}")

    if not is_refusal(answer):
        citations = get_unique_citations(chunks)
        print("Sources:")
        for citation in citations:
            print(f"  - {citation}")
    print()

    history.append((question, answer))