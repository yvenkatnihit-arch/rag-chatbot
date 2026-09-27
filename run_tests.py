from qa_chain import answer_question
from citations import get_unique_citations
from metrics_data import METRICS_TEST_SET

for item in METRICS_TEST_SET:
    question = item["question"]
    answer, chunks = answer_question(question)
    citations = get_unique_citations(chunks)

    print(f"Q: {question}")
    print(f"A: {answer}")
    print(f"Sources: {citations}")
    print("-" * 60)