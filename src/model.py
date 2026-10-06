"""
Backward-compatibility re-export for BiLSTM model architecture.
Primary implementation has been moved to `src/models/bilstm.py`.
"""
from src.models.bilstm import AttentionPooling, SiameseBiLSTMRelevanceModel

__all__ = ["AttentionPooling", "SiameseBiLSTMRelevanceModel"]

