import faiss
import numpy as np


def create_faiss_index(embeddings):

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index



def search_faiss_index(query_embedding, index, chunks, top_k=3):
    
    distances, indices = index.search(query_embedding, top_k)
    results = [(chunks[i], distances[0][pos]) for pos, i in enumerate(indices[0])]
    return results
