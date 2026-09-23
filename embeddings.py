from config import EMBEDDING_MODEL
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Create the embedding model object. It finds GOOGLE_API_KEY automatically, like the chat model did.
embedder = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Three sentences: two are related in meaning, one is unrelated
sentence_a = "Nike's revenue grew significantly last year."
sentence_b = "The company's sales increased a lot in the past year."
sentence_c = "Bill Bowerman was a University of Oregon track coach."
sentence_d = "I had pasta for dinner last night."

# embed_query() turns ONE piece of text into a list of numbers
vector_a = embedder.embed_query(sentence_a)
vector_b = embedder.embed_query(sentence_b)
vector_c = embedder.embed_query(sentence_c)
vector_d = embedder.embed_query(sentence_d)

print(f"Vector length (dimensions): {len(vector_a)}")
print(f"First 5 numbers of vector A: {vector_a[:5]}")


# A simple cosine similarity function, written out so you can see the math
def cosine_similarity(v1, v2):
    dot_product = sum(x * y for x, y in zip(v1, v2))       # multiply matching positions, add them up
    length_v1 = sum(x * x for x in v1) ** 0.5               # length of vector 1
    length_v2 = sum(y * y for y in v2) ** 0.5               # length of vector 2
    return dot_product / (length_v1 * length_v2)             # cosine of the angle between them

similarity_ab = cosine_similarity(vector_a, vector_b)
similarity_ac = cosine_similarity(vector_a, vector_c)
similarity_ad = cosine_similarity(vector_a, vector_d)

print(f"\nSimilarity between A and B (related meaning): {similarity_ab:.4f}")
print(f"Similarity between A and C (unrelated meaning): {similarity_ac:.4f}")
print(f"Similarity between A and D (very unrelated): {similarity_ad:.4f}")