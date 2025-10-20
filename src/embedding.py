"""
Embedding module for generating text embeddings using various models.

This module provides a flexible interface for working with different embedding models,
including OpenAI's API and local transformer models.
"""

import os
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union


class BaseEmbedding(ABC):
    """Abstract base class for embedding models."""

    def __init__(self):
        self.model_name = None

    @abstractmethod
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for the given text(s).

        Args:
            texts: A single string or list of strings to embed.

        Returns:
            numpy array of embeddings with shape (n_texts, embedding_dim)
        """
        pass

    def get_model_name(self) -> str:
        """Return the name of the embedding model."""
        return self.model_name


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI embedding model implementation."""

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model_name: str = "text-embedding-3-large",
        batch_size: int = 100,
    ):
        """
        Initialize OpenAI embedding model.

        Args:
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY env variable)
            base_url: Optional custom base URL for OpenAI API
            model_name: Name of the OpenAI embedding model to use
            batch_size: Number of texts to process in each batch
        """
        super().__init__()
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai is not installed. Please install it with: pip install openai"
            )

        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"), base_url=base_url
        )
        self.model_name = model_name
        self.batch_size = batch_size

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings using OpenAI API."""
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = self.client.embeddings.create(input=batch, model=self.model_name)
            embeddings.extend([item.embedding for item in response.data])

        return np.array(embeddings)


class LocalEmbedding(BaseEmbedding):
    """Local embedding model using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
        """
        Initialize local embedding model.

        Args:
            model_name: Name of the sentence-transformers model to use
            device: Device to run the model on ('cuda', 'cpu', or None for auto)
        """
        super().__init__()
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Please install it with: pip install sentence-transformers"
            )

        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings using local transformer model."""
        if isinstance(texts, str):
            texts = [texts]

        return self.model.encode(texts, show_progress_bar=len(texts) > 100)


class HuggingFaceEmbedding(BaseEmbedding):
    """HuggingFace embedding model implementation."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: str = None,
        normalize_embeddings: bool = True,
    ):
        """
        Initialize HuggingFace embedding model.

        Args:
            model_name: Name of the HuggingFace model to use
            device: Device to run the model on ('cuda', 'cpu', or None for auto)
            normalize_embeddings: Whether to normalize the embeddings
        """
        super().__init__()
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Please install it with: pip install sentence-transformers"
            )

        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings using HuggingFace model."""
        if isinstance(texts, str):
            texts = [texts]

        return self.model.encode(
            texts,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=self.normalize_embeddings,
        )


def create_embedding_model(
    model_type: str = "openai",
    model_name: str = None,
    api_key: str = None,
    base_url: str = None,
    device: str = None,
    **kwargs,
) -> BaseEmbedding:
    """
    Factory function to create an embedding model.

    Args:
        model_type: Type of model ('openai', 'local', 'huggingface')
        model_name: Specific model name (optional, uses defaults if not provided)
        api_key: API key for OpenAI models
        base_url: Base URL for OpenAI API
        device: Device to run local models on
        **kwargs: Additional model-specific arguments

    Returns:
        An instance of BaseEmbedding

    Examples:
        >>> # OpenAI model
        >>> model = create_embedding_model('openai', api_key='sk-...')
        
        >>> # Local model
        >>> model = create_embedding_model('local')
        
        >>> # Custom HuggingFace model
        >>> model = create_embedding_model('huggingface', 
        ...                                 model_name='BAAI/bge-base-en-v1.5')
    """
    model_type = model_type.lower()

    if model_type == "openai":
        model_name = model_name or "text-embedding-3-large"
        return OpenAIEmbedding(
            api_key=api_key, base_url=base_url, model_name=model_name, **kwargs
        )
    elif model_type == "local":
        model_name = model_name or "all-MiniLM-L6-v2"
        return LocalEmbedding(model_name=model_name, device=device, **kwargs)
    elif model_type == "huggingface":
        model_name = model_name or "BAAI/bge-large-en-v1.5"
        return HuggingFaceEmbedding(model_name=model_name, device=device, **kwargs)
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Choose from: 'openai', 'local', 'huggingface'"
        )


# Convenience aliases for backward compatibility
EmbeddingModel = create_embedding_model

