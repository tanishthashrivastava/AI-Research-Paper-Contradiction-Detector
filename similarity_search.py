import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
data = pd.read_csv("data/papers.csv")

model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = model.encode(data['conclusion'].tolist())

query = "Transformer improves translation accuracy"

query_embedding = model.encode([query])

similarities = cosine_similarity(query_embedding, embeddings)

best_match = similarities.argmax()

print("Query:", query)
print("Most Similar Paper:")
print(data.iloc[best_match]['title'])
print(data.iloc[best_match]['conclusion'])