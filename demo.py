import math
import ollama


EMBEDDING_MODEL = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"
LANGUAGE_MODEL = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"

VECTOR_DB = []


def load_dataset(filename):
    """Load non-empty lines from a text file."""
    with open(filename, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def add_chunk_to_database(chunk):
    """Create an embedding for one chunk and store it."""
    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=chunk,
    )

    embedding = response["embeddings"][0]
    VECTOR_DB.append((chunk, embedding))


def cosine_similarity(a, b):
    """Return the cosine similarity between vectors a and b."""

    if len(a) != len(b):
        raise ValueError("Vectors must have the same length.")

    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def retrieve(query, top_n=3):
    """Return the top matching chunks and their similarity scores."""

    query_embedding = ollama.embed(
        model=EMBEDDING_MODEL,
        input=query,
    )["embeddings"][0]

    similarities = []

    for chunk, embedding in VECTOR_DB:
        score = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, score))

    similarities.sort(key=lambda item: item[1], reverse=True)

    return similarities[:top_n]


def select_knowledge_base():
    """Ask the user which knowledge-base file should be loaded."""

    print("Choose a knowledge base:")
    print("1 - cat-facts.txt")
    print("2 - cat-db-fiction-confusing.txt")

    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        return "cat-facts.txt"

    if choice == "2":
        return "cat-db-fiction-confusing.txt"

    raise ValueError("Invalid choice. Please enter 1 or 2.")


def main():
    # Test cosine similarity with simple vectors
    assert math.isclose(cosine_similarity([1, 0], [1, 0]), 1.0)
    assert math.isclose(cosine_similarity([1, 0], [0, 1]), 0.0)
    print("Cosine similarity tests passed.")

    # Select and load the knowledge base
    filename = select_knowledge_base()
    dataset = load_dataset(filename)

    print(f"\nKnowledge base: {filename}")
    print(f"Number of chunks: {len(dataset)}")
    print("First two chunks:")
    print(dataset[:2])

    # Clear the vector database before rebuilding it
    VECTOR_DB.clear()

    # Build the in-memory vector database
    for chunk in dataset:
        add_chunk_to_database(chunk)

    print(f"\nVector DB size: {len(VECTOR_DB)}")

    # Ask the user for a question
    input_query = input("\nAsk a question: ")

    # Retrieve relevant chunks
    retrieved_knowledge = retrieve(input_query, top_n=3)

    print("\nRetrieved knowledge:")
    for chunk, similarity in retrieved_knowledge:
        print(f"- ({similarity:.3f}) {chunk}")

    # Build the context for the language model
    context = "\n".join(
        f"- {chunk}" for chunk, _similarity in retrieved_knowledge
    )

    # Create the grounded prompt
    instruction_prompt = f"""You are a grounded question-answering assistant.

Answer the user's question using only the context below.

Rules:
1. Answer every part of the user's question.
2. Do not use outside knowledge.
3. Do not guess or invent information.
4. Keep records for different animals separate, even when they have the same name.
5. Use location, owner, collar color, dates, or other identifiers to determine which record belongs to which animal.
6. Before answering, check that all facts in your answer refer to the same animal.
7. If two statements about the same animal contradict each other, prefer a clearly dated newer record.
8. If the context directly states the answer, use that answer without unnecessary calculation.
9. If the context does not contain enough evidence, respond exactly:
"The answer is not in the knowledge base."
10. Give a short, direct, and complete answer.

Context:
{context}
""" 

    # Generate and stream the answer
    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {"role": "system", "content": instruction_prompt},
            {"role": "user", "content": input_query},
        ],
        stream=True,
    )

    print("\nAnswer:")
    for response_chunk in stream:
        print(
            response_chunk["message"]["content"],
            end="",
            flush=True,
        )

    print()


if __name__ == "__main__":
    main()