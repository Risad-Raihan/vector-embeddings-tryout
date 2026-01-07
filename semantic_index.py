from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "How to file income tax in Bangladesh",
    "VAT registration process",
    "Corporate tax compliance rules",
    "Penalty for late tax submission",
    "Fine-tuning large language models",
    "Agentic RAG architecture",
    "Deploying AI models on cloud infrastructure",
    "This is a random unrelated sentence about cooking pasta",
]

doc_embeddings = model.encode(documents)

def cosine_sim(a,b):
    return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search(query, top_k = 3, threshold=0.45):
    q_emb = model.encode(query)
    results = []
    
    for emb, doc in zip (doc_embeddings, documents):
        score = cosine_sim(q_emb, emb)
        if score >= threshold:
            results.append((score, doc))

    return sorted(results, reverse=True)[:top_k]

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(doc_embeddings) 

clusters = {}

for label,doc in zip(labels, documents):
    clusters.setdefault(label, []).append(doc)

for cluster_id, docs in clusters.items():
    print(f"\nCluster {cluster_id}")
    for d in docs:
        print("-", d)