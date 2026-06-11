"""Visualización de embeddings vectoriales del RAG local.

Genera tres gráficos a partir del índice FAISS persistido:
  1. rag_embeddings_2d.png    — PCA + t-SNE de los vectores coloreados por archivo.
  2. rag_similarity_heatmap.png — Matriz de similaridad coseno entre chunks.
  3. rag_score_analysis.png     — Distribución de scores + score por página para una query
     de ejemplo.

Uso:
    python visualize_vectors.py [--query "texto de consulta"]

Requiere el virtualenv del proyecto activado y el índice FAISS existente en data/index/.
"""

import argparse
import os
from pathlib import Path

import faiss
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from rag.config import load_config
from rag.embedder import Embedder
from rag.vectorstore import VectorStore


def load_store() -> tuple:
    """Carga la config, el VectorStore y el índice FAISS desde disco.

    Returns:
        tuple: (config, store, index, embeddings, filenames, pages)
            - config:       RAGConfig con la configuración actual.
            - store:        VectorStore con los chunks y metadatos cargados.
            - index:        faiss.Index leído del archivo vectors.index.
            - embeddings:   np.ndarray de forma (N, D) con todos los vectores.
            - filenames:    lista de str con el nombre de archivo de cada chunk.
            - pages:        lista de int con el número de página de cada chunk.
    """
    config = load_config()
    store = VectorStore()
    store.load(config.index_path)

    index_file = Path(config.index_path) / "vectors.index"
    index = faiss.read_index(str(index_file))
    embeddings = index.reconstruct_n(0, index.ntotal)

    filenames = [chunk.metadata.get("filename", "unknown") for chunk in store.chunks]
    pages = [chunk.metadata.get("page", 0) for chunk in store.chunks]

    print(f"Total vectors: {len(embeddings)}")
    print(f"Unique docs: {len(set(filenames))}")

    return config, store, index, embeddings, filenames, pages


def plot_embedding_2d(embeddings: np.ndarray, filenames: list[str]) -> str:
    """Genera un gráfico 2x1 con PCA (izquierda) y t-SNE (derecha) de los embeddings.

    Cada punto se colorea según el archivo de origen del chunk.

    Args:
        embeddings: Matriz (N, D) de vectores de embedding.
        filenames:  Lista de N strings con el nombre de archivo de cada chunk.

    Returns:
        str: Ruta del archivo PNG generado.
    """
    pca = PCA(n_components=2)
    coords_pca = pca.fit_transform(embeddings)
    print(f"\nPCA - Variance: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}")

    perplexity = min(30, len(embeddings) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=500)
    coords_tsne = tsne.fit_transform(embeddings)

    unique_files = list(set(filenames))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_files)))
    file_to_color = {f: colors[i] for i, f in enumerate(unique_files)}

    plt.figure(figsize=(14, 6.5))

    plt.subplot(1, 2, 1)
    for f in unique_files:
        mask = [fn == f for fn in filenames]
        pts = coords_pca[mask]
        plt.scatter(pts[:, 0], pts[:, 1], c=[file_to_color[f]], label=f, alpha=0.6, s=20)
    plt.title("PCA - Embeddings (coloreado por archivo)")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    for f in unique_files:
        mask = [fn == f for fn in filenames]
        pts = coords_tsne[mask]
        plt.scatter(pts[:, 0], pts[:, 1], c=[file_to_color[f]], label=f, alpha=0.6, s=20)
    plt.title("t-SNE - Embeddings (coloreado por archivo)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0.10, 1, 1])
    plt.figtext(
        0.5, 0.01,
        "PCA: Reduce 384 dimensiones a 2D preservando la máxima varianza. "
        "t-SNE: Proyección no-lineal que preserva vecindarios locales (perplexity=30). "
        "Cada color = un documento. Puntos cercanos = chunks semánticamente similares.",
        ha="center", fontsize=7, wrap=True,
    )
    out_path = "rag_embeddings_2d.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def plot_similarity_heatmap(embeddings: np.ndarray, filenames: list[str], pages: list[int]) -> str:
    """Genera una matriz de similaridad coseno entre un subconjunto de chunks.

    Toma una muestra de hasta 100 chunks ordenados por (archivo, página) para
    revelar estructura temática en los embeddings.

    Args:
        embeddings: Matriz (N, D) de vectores de embedding.
        filenames:  Lista de N strings con el nombre de archivo de cada chunk.
        pages:      Lista de N ints con el número de página de cada chunk.

    Returns:
        str: Ruta del archivo PNG generado.
    """
    n_sample = min(100, len(embeddings))
    sorted_idx = sorted(range(len(embeddings)), key=lambda i: (filenames[i], pages[i]))
    sorted_idx = sorted_idx[:n_sample]
    sample_emb = embeddings[sorted_idx]

    norm = sample_emb / np.linalg.norm(sample_emb, axis=1, keepdims=True)
    sim_matrix = norm @ norm.T

    vmin = float(sim_matrix.min())
    vmax = float(sim_matrix.max())
    plt.figure(figsize=(10, 8.5))
    plt.imshow(sim_matrix, cmap="viridis", vmin=vmin, vmax=vmax)
    plt.colorbar(label="Similaridad del Coseno")
    plt.title(f"Matriz de Similaridad - {n_sample} chunks (escala: [{vmin:.2f}, {vmax:.2f}])")
    plt.xlabel("Chunk index")
    plt.ylabel("Chunk index")
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.figtext(
        0.5, 0.01,
        "Similaridad del Coseno = cos(θ) = (A·B) / (||A||·||B||). Rango: [-1, 1].\n"
        "1 = vectores idénticos, 0 = ortogonales (sin relación), <0 = opuestos.\n"
        "Bloques diagonales = chunks del mismo documento (ordenados por archivo y página).",
        ha="center", fontsize=7, wrap=True,
    )
    out_path = "rag_similarity_heatmap.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def plot_score_analysis(
    config,
    index: faiss.Index,
    embeddings: np.ndarray,
    pages: list[int],
    query: str = "framework ETL Coppel",
) -> str:
    """Genera un análisis de scores para una query de ejemplo.

    Dos subgráficos:
      1. Histograma de distribución de scores contra todos los chunks.
      2. Score por página (solo páginas > 0) con línea del threshold.

    Args:
        config:     RAGConfig (aporta similarity_threshold).
        index:      faiss.Index ya cargado.
        embeddings: Matriz (N, D) de vectores (se usa solo para N).
        pages:      Lista de N ints con el número de página.
        query:      Texto de la query de ejemplo (default: "framework ETL Coppel").

    Returns:
        str: Ruta del archivo PNG generado.
    """
    embedder = Embedder(config)
    query_emb = embedder.embed_query(query)
    faiss.normalize_L2(query_emb.reshape(1, -1))

    all_scores, _ = index.search(query_emb.reshape(1, -1), len(embeddings))

    plt.figure(figsize=(12, 4.5))

    plt.subplot(1, 2, 1)
    plt.hist(all_scores[0], bins=30, alpha=0.7, edgecolor="black")
    plt.axvline(
        config.similarity_threshold,
        color="red",
        linestyle="--",
        label=f"threshold={config.similarity_threshold}",
    )
    plt.xlabel("Score (similaridad coseno con la query)")
    plt.ylabel("Cantidad de chunks")
    plt.title(f'Distribución de Scores: "{query}"')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    valid = [(p, s) for p, s in zip(pages, all_scores[0]) if p > 0]
    if valid:
        p_vals, s_vals = zip(*valid)
        plt.scatter(p_vals, s_vals, alpha=0.5, s=10)
        plt.axhline(config.similarity_threshold, color="red", linestyle="--")
        plt.xlabel("Página del documento")
        plt.ylabel("Score")
        plt.title(f'Score por Página: "{query}"')
        plt.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0.15, 1, 1])
    plt.figtext(
        0.5, 0.01,
        "Score = similaridad coseno entre el embedding de la query y el de cada chunk.\n"
        "La línea roja (threshold) filtra chunks irrelevantes: solo los que están a la derecha (histograma) "
        "o arriba (score por página) superan el filtro.\n"
        "Score por página ayuda a identificar qué secciones del documento contienen la información buscada.",
        ha="center", fontsize=7, wrap=True,
    )
    out_path = "rag_score_analysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def main() -> None:
    """Punto de entrada: carga datos y genera los tres gráficos de visualización."""
    parser = argparse.ArgumentParser(description="Visualiza embeddings del RAG local.")
    parser.add_argument(
        "--query",
        type=str,
        default="framework ETL Coppel",
        help='Query de ejemplo para el análisis de scores (default: "framework ETL Coppel")',
    )
    args = parser.parse_args()

    config, store, index, embeddings, filenames, pages = load_store()

    plot_embedding_2d(embeddings, filenames)
    plot_similarity_heatmap(embeddings, filenames, pages)
    plot_score_analysis(config, index, embeddings, pages, query=args.query)

    print(f"\nGenerado en: {os.getcwd()}")
    print("  - rag_embeddings_2d.png")
    print("  - rag_similarity_heatmap.png")
    print("  - rag_score_analysis.png")


if __name__ == "__main__":
    main()
