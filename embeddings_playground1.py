from sentence_transformers import SentenceTransformer
import numpy as np 

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "How do i file income tax in Bangladesh?",
    "Process for submitting tax returns",
    "Best guitar strings for acoustic",
    "Neural networks for image recognition"
]

embeddings = model.encode(sentences)

def cosine_sim(a,b):
    return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))

query = "How to submit tax documents?"
query_emb = model.encode(query)

for i,s in enumerate(sentences):
    score = cosine_sim(query_emb, embeddings[i])
    print(f"{score:<.3f} → {s}")
