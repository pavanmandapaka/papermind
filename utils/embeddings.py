from sentence_transformers import SentenceTransformer

def get_embedding_model():
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return model

def create_embeddings(chunks, model):

    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings