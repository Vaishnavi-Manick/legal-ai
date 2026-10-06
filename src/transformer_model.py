"""
Backward-compatibility re-export for Transformer model architecture.
Primary implementation has been moved to `src/models/transformer.py`.
"""
from src.models.transformer import LegalTransformerCrossEncoder

__all__ = ["LegalTransformerCrossEncoder"]

