from datetime import datetime
from database.vector_store import VectorStore
from services.synthesizer import Synthesizer
from timescale_vector import client

# Initialize VectorStore
vec = VectorStore()


def run_query(label: str, question: str, **search_kwargs):
    """Helper to run a query, print the label, answer, thought process and context flag."""
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"QUESTION: {question}")
    print("="*60)
    results = vec.search(question, limit=3, **search_kwargs)
    response = Synthesizer.generate_response(question=question, context=results)
    print(f"\nANSWER: {response.answer}")
    print("\nThought process:")
    for thought in response.thought_process:
        print(f"  - {thought}")
    print(f"\nEnough context: {response.enough_context}")


# ==============================================================
# SECTION 1 — ORIGINAL FAQ TESTS (commented out temporarily)
# Restore these when table_name is reverted back to "embeddings"
# ==============================================================

# # --------------------------------------------------------------
# # Shipping question
# # --------------------------------------------------------------
# relevant_question = "What are your shipping options?"
# results = vec.search(relevant_question, limit=3)
# response = Synthesizer.generate_response(question=relevant_question, context=results)
# print(f"\n{response.answer}")
# print("\nThought process:")
# for thought in response.thought_process:
#     print(f"- {thought}")
# print(f"\nContext: {response.enough_context}")

# # --------------------------------------------------------------
# # Irrelevant question
# # --------------------------------------------------------------
# irrelevant_question = "What is the weather in Tokyo?"
# results = vec.search(irrelevant_question, limit=3)
# response = Synthesizer.generate_response(question=irrelevant_question, context=results)
# print(f"\n{response.answer}")
# print("\nThought process:")
# for thought in response.thought_process:
#     print(f"- {thought}")
# print(f"\nContext: {response.enough_context}")

# # --------------------------------------------------------------
# # Metadata filtering
# # --------------------------------------------------------------
# metadata_filter = {"category": "Shipping"}
# results = vec.search(relevant_question, limit=3, metadata_filter=metadata_filter)
# response = Synthesizer.generate_response(question=relevant_question, context=results)
# print(f"\n{response.answer}")
# print("\nThought process:")
# for thought in response.thought_process:
#     print(f"- {thought}")
# print(f"\nContext: {response.enough_context}")

# # --------------------------------------------------------------
# # Advanced filtering using Predicates
# # --------------------------------------------------------------
# predicates = client.Predicates("category", "==", "Shipping")
# results = vec.search(relevant_question, limit=3, predicates=predicates)

# predicates = client.Predicates("category", "==", "Shipping") | client.Predicates(
#     "category", "==", "Services"
# )
# results = vec.search(relevant_question, limit=3, predicates=predicates)

# predicates = client.Predicates("category", "==", "Shipping") & client.Predicates(
#     "created_at", ">", "2024-09-01"
# )
# results = vec.search(relevant_question, limit=3, predicates=predicates)

# # --------------------------------------------------------------
# # Time-based filtering
# # --------------------------------------------------------------
# # September — Returning results
# time_range = (datetime(2024, 9, 1), datetime(2024, 9, 30))
# results = vec.search(relevant_question, limit=3, time_range=time_range)

# # August — Not returning any results
# time_range = (datetime(2024, 8, 1), datetime(2024, 8, 30))
# results = vec.search(relevant_question, limit=3, time_range=time_range)


# ==============================================================
# SECTION 2 — DOCUMENT PIPELINE TESTS (Docling paper)
# Active while table_name = "document_embeddings"
# ==============================================================

# --------------------------------------------------------------
# Happy path — questions with clear answers in the document
# --------------------------------------------------------------

run_query(
    label="Happy Path 1 — Architecture question",
    question="What is the Docling architecture?",
)

run_query(
    label="Happy Path 2 — Specific feature question",
    question="How does Docling handle PDF parsing?",
)

run_query(
    label="Happy Path 3 — Model question",
    question="What machine learning models does Docling use?",
)

# --------------------------------------------------------------
# Unhappy path — questions with no relevant answer in the document
# --------------------------------------------------------------

run_query(
    label="Unhappy Path 1 — Completely irrelevant question",
    question="What is the weather in Tokyo?",
)

run_query(
    label="Unhappy Path 2 — Related domain but not in document",
    question="How do I integrate Docling with LangChain?",
)

run_query(
    label="Unhappy Path 3 — Vague / ambiguous question",
    question="Tell me everything.",
)

run_query(
    label="Unhappy Path 4 — Empty-like question",
    question="?",
)

run_query(
    label="Unhappy Path 5 — Question about a different paper",
    question="What does the GPT-4 technical report say about safety?",
)
