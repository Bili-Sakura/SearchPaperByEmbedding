"""
SearchPaperByEmbedding - Semantic search for academic papers.
"""

from .search import PaperSearcher
from .embedding import (
    BaseEmbedding,
    OpenAIEmbedding,
    LocalEmbedding,
    HuggingFaceEmbedding,
    create_embedding_model,
)

__all__ = [
    "PaperSearcher",
    "BaseEmbedding",
    "OpenAIEmbedding",
    "LocalEmbedding",
    "HuggingFaceEmbedding",
    "create_embedding_model",
]

