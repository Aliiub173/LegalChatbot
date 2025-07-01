from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

model = SentenceTransformer("all-MiniLM-L6-v2")

def build_index(chunks, save_dir="data"):
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    faiss.write_index(index, f"{save_dir}/divorce_index.faiss")
    with open(f"{save_dir}/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

def load_index_and_chunks(data_dir="data"):
    index = faiss.read_index(f"{data_dir}/divorce_index.faiss")
    with open(f"{data_dir}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks, model