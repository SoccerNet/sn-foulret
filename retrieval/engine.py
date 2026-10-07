"""FAISS-backed retrieval engine shared by all embedding models."""

import os
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import faiss
import torch
from typing import List, Dict, Union, Optional

from retrieval.models.embedder_base import BaseEmbeddingModel

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class RetrievalEngine:
    """
    Model-agnostic retrieval engine.

    Manages the FAISS index and metadata. Delegates all embedding computation
    to a BaseEmbeddingModel instance, so it works with any model backend
    (Qwen3-VL-Embedding, etc.).

    """

    def __init__(
        self,
        model: Optional[BaseEmbeddingModel],
        project_root: str,
        embedding_path: str,
        index_path: str,
    ):
        """
        Args:
            model: An instance of BaseEmbeddingModel.
            project_root: Root directory of the project, used to resolve relative paths.
            embedding_path: Path to the embeddings metadata JSON file.
            index_path: Path to the FAISS index binary file.
        """
        self.model = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.project_root = project_root
        self.embedding_path = embedding_path
        self.index_path = index_path

        self.df = None
        self.index = None

        self._load_if_exists()

    def _load_if_exists(self):
        """Loads the index and metadata if they exist."""
        if os.path.exists(self.embedding_path):
            logger.info(f"Loading existing embeddings from {self.embedding_path}...")
            self.df = pd.read_json(self.embedding_path)
            # Ensure 'path' column is treated as string, even if some values were None before
            self.df['path'] = self.df['path'].astype(str)
            
            if os.path.exists(self.index_path):
                logger.info(f"Loading FAISS index from {self.index_path}...")
                self.index = self._load_faiss_index()
            else:
                logger.warning("No FAISS index found at %s.", self.index_path)
        else:
            logger.warning("No saved embeddings found. Call 'build_index_from_data' to create them.")

    def _load_faiss_index(self):
        """Loads the FAISS index onto the GPU if available."""
        cpu_index = faiss.read_index(self.index_path)
        if self.device == "cuda":
            try:
                res = faiss.StandardGpuResources()
                gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                logger.info(f"FAISS index loaded with {gpu_index.ntotal} vectors on GPU.")
                return gpu_index
            except Exception as e:
                logger.error(f"Failed to load FAISS index to GPU, falling back to CPU. Error: {e}")
        logger.info(f"FAISS index loaded with {cpu_index.ntotal} vectors on CPU.")
        return cpu_index

    def embed_video_from_path(self, relative_action_dir: str, is_query: bool = True) -> Union[np.ndarray, None]:
        """
        Generates a normalized embedding for a video.

        Resolves relative paths against project_root, then delegates to the model.
        """
        if os.path.isabs(relative_action_dir):
            absolute_path = relative_action_dir
        else:
            absolute_path = os.path.join(self.project_root, relative_action_dir)
        return self.model.embed_video_from_path(absolute_path, is_query=is_query)

    def search(self, query_emb: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Performs a search on the FAISS index."""
        if self.index is None or self.df is None:
            logger.error("Index not built or loaded. Cannot perform search.")
            return []

        faiss.normalize_L2(query_emb)
        distances, indices = self.index.search(query_emb, top_k)

        results = self.df.iloc[indices[0]].copy()
        results["score"] = distances[0]

        return results.to_dict(orient="records")

    def search_by_video(self, relative_video_path: str, top_k: int = 5) -> List[Dict]:
        """Convenience method to search using a relative video path."""
        logger.info(f"Searching for video: '{relative_video_path}'")
        query_emb = self.embed_video_from_path(relative_video_path)
        if query_emb is None:
            return []
        return self.search(query_emb, top_k)

    def build_index_from_data(self, entries: List[Dict]):
        """Builds embeddings and the FAISS index from the prepared data entries."""
        logger.info("Building embeddings and FAISS index from scratch...")
        vectors = []
        metadata = []

        for entry in tqdm(entries, desc="Generating Embeddings"):
            if entry["type"] != "video":
                logger.warning(f"Unknown entry type: {entry['type']}")
                continue
            emb = self.embed_video_from_path(entry["path"], is_query=False)

            if emb is not None:
                vectors.append(emb[0])
                metadata.append(entry)

        if not vectors:
            logger.error("No valid embeddings were generated. Halting index build.")
            return False

        embeddings = np.vstack(vectors).astype("float32")
        # L2-normalize in-place (faiss.normalize_L2 may not exist in all builds)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings /= norms

        # Build index on CPU first
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        # Save metadata and CPU index
        self.df = pd.DataFrame(metadata)
        self.df.to_json(self.embedding_path, orient="records", indent=2)
        logger.info(f"Saved metadata to {self.embedding_path}")

        faiss.write_index(index, self.index_path)
        logger.info(f"Saved FAISS index to {self.index_path}")

        # Finally, load the created index for immediate use
        self._load_if_exists()
        return True
