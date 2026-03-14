import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

print("Loading dataset...")

data = pd.read_csv("data/papers.csv")

model = SentenceTransformer('all-MiniLM-L6-v2')

print("Generating embeddings...")

texts = data["conclusion"].tolist()
embeddings = model.encode(texts)

dimension = embeddings.shape[1]

print("Creating FAISS index...")

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

print("Total vectors stored:", index.ntotal)

# Save index
faiss.write_index(index, "paper_index.faiss")

# Save metadata
with open("metadata.pkl", "wb") as f:
    pickle.dump(data, f)

print("Vector database created successfully!")