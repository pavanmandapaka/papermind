# test_embeddings_faiss.py

from utils.embeddings import get_embedding_model, create_embeddings
from utils.vector_store import create_faiss_index, search_faiss_index

# Sample text chunks (pretend from your notes)
chunks = [
    "Machine learning is a subset of artificial intelligence.",
    "Supervised learning uses labeled data to train models.",
    "Unsupervised learning finds hidden patterns in data."
]

print("Loading model...")
model = get_embedding_model()

print("Creating embeddings...")
embeddings = create_embeddings(chunks, model)
print("Embeddings shape:", embeddings.shape)

print("Creating FAISS index...")
index = create_faiss_index(embeddings)

# Example query
query = "What type of learning uses labeled data?"
query_emb = model.encode([query])
results = search_faiss_index(query_emb, index, chunks, top_k=2)

print("\nTop results:")
for r in results:
    print("Chunk:", r[0])
    print("Distance:", r[1])
    print("------")
