"""
Evaluation Module for Legal AI Relevance Models.
Contains TREC IR metric calculations, prediction saving, and visualization tools.
"""

from src.utils import (
    calculate_metrics,
    save_metrics_csv,
    save_predictions_csv,
    save_history_csv,
    save_loss_plot,
    DiagnosticsLogger,
    compute_grad_norm
)

__all__ = [
    "calculate_metrics",
    "save_metrics_csv",
    "save_predictions_csv",
    "save_history_csv",
    "save_loss_plot",
    "DiagnosticsLogger",
    "compute_grad_norm"
]

