import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

# Load vector index
index = faiss.read_index("paper_index.faiss")

# Load metadata
with open("metadata.pkl", "rb") as f:
    data = pickle.load(f)

model = SentenceTransformer('all-MiniLM-L6-v2')

query = input("Enter research query: ")

query_vector = model.encode([query])

D, I = index.search(np.array(query_vector), k=3)

print("\nTop Similar Papers:\n")

for i in I[0]:
    print("Title:", data.iloc[i]["title"])
    print("Conclusion:", data.iloc[i]["conclusion"])
    print("-----")