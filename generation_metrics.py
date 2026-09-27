from qa_chain import answer_question, format_chunks_as_context
from metrics_data import METRICS_TEST_SET
from faithfulness import check_faithfulness, extract_score
from correctness import check_correctness

total_faithfulness = 0
correct_count = 0

for item in METRICS_TEST_SET:
    question = item["question"]
    expected_answer = item["expected_answer"]

    answer, chunks = answer_question(question)
    context = format_chunks_as_context(chunks)

    faithfulness_result = check_faithfulness(answer, context)
    faithfulness_score = extract_score(faithfulness_result)
    if faithfulness_score is not None:
        total_faithfulness += faithfulness_score

    correctness_result = check_correctness(answer, expected_answer)

    print(f"Q: {question}")
    print(f"Generated: {answer}")
    print(f"Expected: {expected_answer}")
    print(f"Faithfulness: {faithfulness_result}")
    print(f"Correctness: {correctness_result}")

    if "Verdict: CORRECT" in correctness_result:
        correct_count += 1

    print("-" * 60)

num_questions = len(METRICS_TEST_SET)
print(f"\nAverage Faithfulness: {total_faithfulness / num_questions:.2f}")
print(f"Correctness accuracy: {correct_count}/{num_questions} ({correct_count / num_questions * 100:.0f}%)")