from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE 
import matplotlib.pyplot as plt 


#load model
model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    # Tax / Legal
    "How to file income tax in Bangladesh",
    "VAT registration process",
    "Corporate tax compliance rules",
    "Penalty for late tax submission",

    # AI / ML
    "Fine-tuning large language models",
    "Training neural networks",
    "Agentic RAG architecture",
    "Deploying AI models on cloud",

    # Music / Guitar
    "Best acoustic guitar strings",
    "Fingerstyle guitar techniques",
    "Beginner guitar chords",
    "How to tune a guitar",
]

#create embedding
embeddings = model.encode(sentences)

tsne = TSNE(
    n_components = 2,
    perplexity = 4,
    random_state = 42,
    init = "random"
)
embeddings_2d = tsne.fit_transform(embeddings)

#plot
plt.figure(figsize=(10,7))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])

#annotate points
for i, sentence in enumerate(sentences):
    plt.annotate(sentence, (embeddings_2d[i, 0], embeddings_2d[i,1]))


plt.title("2D visualization of Semantic Embeddings")
plt.show()