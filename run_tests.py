from qa_chain import answer_question
from citations import get_unique_citations
from test_questions import TEST_QUESTIONS

for question in TEST_QUESTIONS:
    answer, chunks = answer_question(question)
    citations = get_unique_citations(chunks)

    print(f"Q: {question}")
    print(f"A: {answer}")
    print(f"Sources: {citations}")
    print("-" * 60)