"""
Abstract base class for embedding models.

All model backends (Qwen3-VL-Embedding, etc.) must implement this interface
so the retrieval engine can use them interchangeably.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Union


class BaseEmbeddingModel(ABC):
    """Abstract interface for multimodal embedding models."""

    @abstractmethod
    def embed_text(self, text: str, is_query: bool = True) -> np.ndarray:
        """
        Generate a normalized embedding for a text string.

        Args:
            text: Input text string.
            is_query: If True, use task-specific query instruction.
                      If False, use default document instruction.

        Returns:
            Normalized embedding as float32 numpy array of shape (1, embedding_dim).
        """
        ...

    @abstractmethod
    def embed_video_from_path(self, video_path: str, is_query: bool = True) -> Union[np.ndarray, None]:
        """
        Generate a normalized embedding for a video given its path.

        The path semantics depend on the model:
        - Qwen3-VL: path to a directory containing clip_*.mp4

        Args:
            video_path: Absolute path to the video data (directory or file).
            is_query: If True, use task-specific query instruction.
                      If False, use default document instruction.

        Returns:
            Normalized embedding as float32 numpy array of shape (1, embedding_dim),
            or None if embedding could not be generated.
        """
        ...

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        ...
