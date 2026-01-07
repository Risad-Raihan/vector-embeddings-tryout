from sentence_transformers import SentenceTransformer
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

results = search("How do I submit tax returns?")
for score, doc in results:
    print(f"{score:.3f} → {doc}")

print(search("tax filing procedure"))
print(search("AI deployment"))
print(search("guitar tuning"))