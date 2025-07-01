from rag.embedder import load_index_and_chunks
import numpy as np

index, chunks, model = load_index_and_chunks()

def retrieve_context(query, top_k=3):
    query_vector = model.encode([query])
    D, I = index.search(np.array(query_vector), top_k)
    return " ".join([chunks[i] for i in I[0]])