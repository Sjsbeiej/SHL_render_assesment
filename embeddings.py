"""
Embedding pipeline: Build FAISS index from scraped SHL assessment data.
Uses LangChain's OpenAIEmbeddings for embedding generation.
"""

import json
import os
import pickle

import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings

import config

# Singleton embeddings model
_embeddings_model = None


def get_embeddings_model() -> OpenAIEmbeddings:
    """Get or create the LangChain OpenAIEmbeddings model."""
    global _embeddings_model
    if _embeddings_model is None:
        _embeddings_model = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            api_key=config.OPENAI_API_KEY,
        )
    return _embeddings_model


def load_assessments() -> list[dict]:
    """Load scraped assessments from JSON."""
    with open(config.ASSESSMENTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def build_text_representation(assessment: dict) -> str:
    """Create natural language text representation for embedding.
    Natural sentences embed better than labeled key-value pairs."""
    name = assessment['name']
    desc = assessment.get("description", "")
    types = ', '.join(assessment.get("test_types", []))
    levels = assessment.get("job_levels", "")
    duration = assessment.get("duration_minutes")

    text = f"{name} is a {types} assessment." if types else f"{name}."

    if desc:
        text += f" {desc}"

    if levels:
        text += f" Suitable for {levels} roles."

    if duration:
        text += f" Takes {duration} minutes to complete."

    return text


def get_embeddings(texts: list[str]) -> np.ndarray:
    """Get embeddings using LangChain OpenAIEmbeddings."""
    model = get_embeddings_model()
    all_embeddings = []
    batch_size = 100

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = model.embed_documents(batch)
        all_embeddings.extend(batch_embeddings)
        print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)}")

    return np.array(all_embeddings, dtype="float32")


def embed_query(text: str) -> np.ndarray:
    """Embed a single query using LangChain OpenAIEmbeddings."""
    model = get_embeddings_model()
    embedding = model.embed_query(text)
    return np.array([embedding], dtype="float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build a FAISS index using inner product (cosine similarity after normalization)."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = embeddings / norms

    dimension = normalized.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(normalized)
    return index


def save_index(index: faiss.IndexFlatIP, assessments: list[dict], texts: list[str]):
    """Save FAISS index and metadata."""
    os.makedirs(config.FAISS_INDEX_DIR, exist_ok=True)

    faiss.write_index(index, os.path.join(config.FAISS_INDEX_DIR, "index.faiss"))

    metadata = {
        "assessments": assessments,
        "texts": texts,
    }
    with open(os.path.join(config.FAISS_INDEX_DIR, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    print(f"Saved FAISS index ({index.ntotal} vectors) to {config.FAISS_INDEX_DIR}")


def load_index():
    """Load FAISS index and metadata."""
    index = faiss.read_index(os.path.join(config.FAISS_INDEX_DIR, "index.faiss"))

    with open(os.path.join(config.FAISS_INDEX_DIR, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)

    return index, metadata["assessments"], metadata["texts"]


def main():
    print("Loading assessments...")
    assessments = load_assessments()
    print(f"Loaded {len(assessments)} assessments")

    print("\nBuilding text representations...")
    texts = [build_text_representation(a) for a in assessments]

    print(f"\nSample text representation:\n  {texts[0]}\n")

    print("Generating embeddings via LangChain OpenAIEmbeddings...")
    embeddings = get_embeddings(texts)
    print(f"Embeddings shape: {embeddings.shape}")

    print("\nBuilding FAISS index...")
    index = build_faiss_index(embeddings)

    print("Saving index and metadata...")
    save_index(index, assessments, texts)

    # Quick sanity check
    print("\n=== Sanity Check ===")
    test_query = "Python programming skills test"
    q_embedding = embed_query(test_query)
    q_norm = q_embedding / np.linalg.norm(q_embedding, axis=1, keepdims=True)
    scores, indices = index.search(q_norm, 5)
    print(f"Query: '{test_query}'")
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        print(f"  {i + 1}. [{score:.4f}] {assessments[idx]['name']} - {assessments[idx]['url']}")


if __name__ == "__main__":
    main()
