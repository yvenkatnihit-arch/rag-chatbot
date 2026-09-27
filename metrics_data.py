# For each test question, we manually note:
# - relevant_sources: which document(s) SHOULD be retrieved
# - expected_answer: a short, known-correct answer, used for the correctness metric

METRICS_TEST_SET = [
    # --- Nike ---
    {
        "question": "How many countries does Nike sell its products in?",
        "relevant_sources": ["nike-growth-story.pdf", "nke-10k-2023.pdf"],
        "expected_answer": "170 countries",
    },
    {
        "question": "What brands are part of Nike's portfolio besides Nike and Jordan?",
        "relevant_sources": ["nike-growth-story.pdf"],
        "expected_answer": "Cole Haan, Converse, Hurley, Nike Golf, and Umbro",
    },
    {
        "question": "What is Nike's mission statement?",
        "relevant_sources": ["nike-growth-story.pdf"],
        "expected_answer": "To bring inspiration and innovation to every athlete in the world",
    },
    {
        "question": "What percentage of Nike's revenue has come from outside the U.S. since 2005?",
        "relevant_sources": ["nike-growth-story.pdf", "nke-10k-2023.pdf"],
        "expected_answer": "more than 50 percent",
    },

    # --- Your project report ---
    {
        "question": "What are the system and software requirements in the project?",
        "relevant_sources": ["Final_Venkat_9.pdf"],
        "expected_answer": "Windows or Linux OS, Python, VS Code or PyCharm, Flask and React.js, an LLM via API, SQLite or MySQL, and a Chrome or Edge browser",
    },
    {
        "question": "What are the performance metrics used?",
        "relevant_sources": ["Final_Venkat_9.pdf"],
        "expected_answer": "Precision, Recall, F1-Score, support, confidence scores, and severity classification accuracy",
    },
    {
        "question": "Explain the limitations and challenges of the system.",
        "relevant_sources": ["Final_Venkat_9.pdf"],
        "expected_answer": "Cannot detect sarcasm, depends on an external LLM API, English only, uses an in-memory database that loses data on restart, lacks user authentication, and has performance issues at scale",
    },
    {
        "question": "Explain the methodology, design, and proposed algorithms.",
        "relevant_sources": ["Final_Venkat_9.pdf"],
        "expected_answer": "Uses prompt engineering with a pre-trained LLM to classify cyberbullying, assign severity and confidence, then a rule-based point system triggers warnings and a 24-hour auto-block",
    },

    # --- Cars of the World ---
    {
        "question": "What years was the Ford Model T built, and who made it?",
        "relevant_sources": ["Cars_of_the_World.pdf"],
        "expected_answer": "Built 1908 to 1927 by Ford Motor Company",
    },
    {
        "question": "What is the power output of the Bugatti Veyron?",
        "relevant_sources": ["Cars_of_the_World.pdf"],
        "expected_answer": "1,001 PS",
    },
    {
        "question": "How many Willys MB Jeeps were built during WWII?",
        "relevant_sources": ["Cars_of_the_World.pdf"],
        "expected_answer": "About 640,000, combining Willys and Ford production",
    },
    {
        "question": "What color were all the 1953 Chevrolet Corvettes painted?",
        "relevant_sources": ["Cars_of_the_World.pdf"],
        "expected_answer": "Polo White with a red interior",
    },

    # --- Cross-topic trap ---
    {
        "question": "What is the capital of France?",
        "relevant_sources": [],
        "expected_answer": "I don't know based on the provided documents",   # correct behavior is refusal, not "Paris"
    },
        # --- Exact-term queries: the case hybrid search is actually designed for ---
    {
        "question": "Who is associated with roll number 24MC201A96?",
        "relevant_sources": ["Final_Venkat_9.pdf"],
        "expected_answer": "Y. Venkat",
    },
    {
        "question": "What is the GPW?",
        "relevant_sources": ["Cars_of_the_World.pdf"],
        "expected_answer": "The Ford-built version of the Willys MB Jeep",
    },
]