import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from rag.config import load_config
from rag.vectorstore import VectorStore
from rag.embedder import Embedder
import faiss

config = load_config()
store = VectorStore()
store.load(config.index_path)

# Get embeddings directly from the FAISS index
index_file = Path(config.index_path) / "vectors.index"
index = faiss.read_index(str(index_file))
embeddings = index.reconstruct_n(0, index.ntotal)
texts = [chunk.content[:80] for chunk in store.chunks]
filenames = [chunk.metadata.get("filename", "unknown") for chunk in store.chunks]
pages = [chunk.metadata.get("page", 0) for chunk in store.chunks]

print(f"Total vectors: {len(embeddings)}")
print(f"Unique docs: {len(set(filenames))}")

# --- 2D PCA ---
pca = PCA(n_components=2)
coords_pca = pca.fit_transform(embeddings)
print(f"\nPCA - Variance: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}")

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
unique_files = list(set(filenames))
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_files)))
file_to_color = {f: colors[i] for i, f in enumerate(unique_files)}
for f in unique_files:
    mask = [fn == f for fn in filenames]
    pts = coords_pca[mask]
    plt.scatter(pts[:, 0], pts[:, 1], c=[file_to_color[f]], label=f, alpha=0.6, s=20)
plt.title("PCA - Embeddings (coloreado por archivo)")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.grid(alpha=0.3)

# --- 2D t-SNE ---
perplexity = min(30, len(embeddings) - 1)
tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=500)
coords_tsne = tsne.fit_transform(embeddings)

plt.subplot(1, 2, 2)
for f in unique_files:
    mask = [fn == f for fn in filenames]
    pts = coords_tsne[mask]
    plt.scatter(pts[:, 0], pts[:, 1], c=[file_to_color[f]], label=f, alpha=0.6, s=20)
plt.title("t-SNE - Embeddings (coloreado por archivo)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("rag_embeddings_2d.png", dpi=150, bbox_inches="tight")
print("\nSaved: rag_embeddings_2d.png")

# --- Similarity Heatmap ---
n_sample = min(100, len(embeddings))
sorted_idx = sorted(range(len(store.chunks)), key=lambda i: (filenames[i], pages[i]))
sorted_idx = sorted_idx[:n_sample]
sample_emb = embeddings[sorted_idx]
norm = sample_emb / np.linalg.norm(sample_emb, axis=1, keepdims=True)
sim_matrix = norm @ norm.T

plt.figure(figsize=(10, 8))
plt.imshow(sim_matrix, cmap="viridis", vmin=0, vmax=1)
plt.colorbar(label="Similaridad del Coseno")
plt.title(f"Matriz de Similaridad - {n_sample} chunks")
plt.xlabel("Chunk index")
plt.ylabel("Chunk index")
plt.savefig("rag_similarity_heatmap.png", dpi=150, bbox_inches="tight")
print("Saved: rag_similarity_heatmap.png")

# --- Score Distribution for a sample query ---
embedder = Embedder(config)
query = "framework ETL Coppel"
query_emb = embedder.embed_query(query)
faiss.normalize_L2(query_emb.reshape(1, -1))

all_scores, _ = index.search(query_emb.reshape(1, -1), len(embeddings))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(all_scores[0], bins=30, alpha=0.7, edgecolor="black")
plt.axvline(config.similarity_threshold, color="red", linestyle="--", label=f"threshold={config.similarity_threshold}")
plt.xlabel("Score")
plt.ylabel("Cantidad de chunks")
plt.title(f'Distribucion de Scores: "{query}"')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
valid = [(p, s) for p, s in zip(pages, all_scores[0]) if p > 0]
if valid:
    p_vals, s_vals = zip(*valid)
    plt.scatter(p_vals, s_vals, alpha=0.5, s=10)
    plt.axhline(config.similarity_threshold, color="red", linestyle="--")
    plt.xlabel("Pagina")
    plt.ylabel("Score")
    plt.title(f'Score por Pagina: "{query}"')
    plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("rag_score_analysis.png", dpi=150, bbox_inches="tight")
print("Saved: rag_score_analysis.png")

print("\nGenerado en:", os.getcwd())
print("  - rag_embeddings_2d.png")
print("  - rag_similarity_heatmap.png")
print("  - rag_score_analysis.png")
