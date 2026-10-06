import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Tuple, Set

class DiagnosticsLogger:
    """Logs raw score distributions, gradient norms, and relative parameter updates during training."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.pos_scores = []
        self.neg_scores = []

    def update_scores(self, pos_sims: torch.Tensor, neg_sims: torch.Tensor):
        self.pos_scores.extend(pos_sims.detach().cpu().numpy().flatten().tolist())
        self.neg_scores.extend(neg_sims.detach().cpu().numpy().flatten().tolist())

    def get_score_stats(self) -> Dict[str, float]:
        if not self.pos_scores or not self.neg_scores:
            return {}
        pos_arr = np.array(self.pos_scores)
        neg_arr = np.array(self.neg_scores)
        return {
            "pos_mean": float(np.mean(pos_arr)),
            "pos_std": float(np.std(pos_arr)),
            "neg_mean": float(np.mean(neg_arr)),
            "neg_std": float(np.std(neg_arr)),
            "margin": float(np.mean(pos_arr) - np.mean(neg_arr))
        }

def compute_grad_norm(model: torch.nn.Module) -> float:
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def calculate_metrics(
    rankings: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, Set[str]],
    k_list: List[int] = [5, 10, 20]
) -> Dict[str, float]:
    """
    Computes standard TREC IR metrics: MAP, MRR, Recall@K, nDCG@K, Precision@K.
    rankings: qid -> list of (doc_id, score) sorted descending
    qrels: qid -> set of relevant doc_ids
    """
    map_sum = 0.0
    mrr_sum = 0.0
    recall_sums = {k: 0.0 for k in k_list}
    ndcg_sums = {k: 0.0 for k in k_list}
    prec_sums = {k: 0.0 for k in k_list}
    num_queries = 0

    for qid, ranked_docs in rankings.items():
        rel_set = qrels.get(qid, set())
        if not rel_set:
            continue
            
        num_queries += 1
        retrieved_ids = [doc_id for doc_id, score in ranked_docs]
        
        # 1. MRR & AP
        first_rel_rank = 0
        num_rel_found = 0
        ap_sum = 0.0
        
        for rank_idx, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in rel_set:
                if first_rel_rank == 0:
                    first_rel_rank = rank_idx
                num_rel_found += 1
                ap_sum += num_rel_found / rank_idx
                
        ap = ap_sum / len(rel_set)
        mrr = 1.0 / first_rel_rank if first_rel_rank > 0 else 0.0
        
        map_sum += ap
        mrr_sum += mrr
        
        # 2. Recall@K, nDCG@K, Precision@K
        for k in k_list:
            top_k_ids = retrieved_ids[:k]
            hits = sum(1 for doc_id in top_k_ids if doc_id in rel_set)
            
            # Recall@K & Precision@K
            recall_sums[k] += hits / len(rel_set)
            prec_sums[k] += hits / k
            
            # nDCG@K
            dcg = 0.0
            for i, doc_id in enumerate(top_k_ids, start=1):
                rel = 1.0 if doc_id in rel_set else 0.0
                dcg += rel / math.log2(i + 1)
                
            idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(rel_set), k) + 1))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_sums[k] += ndcg

    if num_queries == 0:
        return {}

    metrics = {
        "MAP": map_sum / num_queries,
        "MRR": mrr_sum / num_queries
    }
    for k in k_list:
        metrics[f"Recall@{k}"] = recall_sums[k] / num_queries
        metrics[f"nDCG@{k}"] = ndcg_sums[k] / num_queries
        metrics[f"P@{k}"] = prec_sums[k] / num_queries
        
    return metrics

def save_loss_plot(history: List[Dict[str, float]], save_path: str = "bilstm_loss_plot.png"):
    """Generates and saves training vs validation loss curve plot."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    
    plt.figure(figsize=(9, 5), dpi=300)
    plt.plot(epochs, train_loss, label="Training Loss", color="#1f77b4", linewidth=2.5, marker="o", markersize=4)
    plt.plot(epochs, val_loss, label="Validation Loss", color="#ff7f0e", linewidth=2.5, linestyle="--", marker="s", markersize=4)
    plt.title("BiLSTM Training vs Validation Loss Across Epochs", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("BCE Loss", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11, frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved loss plot to {save_path}")

def save_metrics_csv(metrics: Dict[str, float], model_name: str = "BiLSTM", save_path: str = "bilstm_metrics.csv"):
    """Saves final IR evaluation metrics to CSV."""
    df = pd.DataFrame([{
        "Model": model_name,
        "P@5": round(metrics.get("P@5", 0.0), 4),
        "Recall@5": round(metrics.get("Recall@5", 0.0), 4),
        "MRR": round(metrics.get("MRR", 0.0), 4),
        "MAP": round(metrics.get("MAP", 0.0), 4),
        "NDCG@5": round(metrics.get("nDCG@5", 0.0), 4),
        "NDCG@10": round(metrics.get("nDCG@10", 0.0), 4)
    }])
    df.to_csv(save_path, index=False)
    print(f"Saved test evaluation metrics to {save_path}")

def save_predictions_csv(rankings: Dict[str, List[Tuple[str, float]]], qrels: Dict[str, Set[str]], save_path: str = "bilstm_test_predictions.csv"):
    """Saves detailed test set ranking predictions to CSV."""
    rows = []
    for qid, ranked_docs in rankings.items():
        rel_set = qrels.get(qid, set())
        for rank, (doc_id, score) in enumerate(ranked_docs, start=1):
            rows.append({
                "query_id": qid,
                "document_id": doc_id,
                "predicted_score": round(float(score), 6),
                "rank": rank,
                "relevance_label": 1 if doc_id in rel_set else 0
            })
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"Saved test predictions to {save_path}")

def save_history_csv(history: List[Dict[str, float]], save_path: str = "bilstm_training_history.csv"):
    """Saves per-epoch training and validation history to CSV."""
    df = pd.DataFrame(history)
    df.to_csv(save_path, index=False)
    print(f"Saved training history to {save_path}")

