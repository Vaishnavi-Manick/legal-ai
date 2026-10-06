import torch
import torch.nn as nn
import torch.nn.functional as F

def get_relevance_loss() -> nn.Module:
    """Returns Binary Cross Entropy with Logits Loss for supervised relevance modeling."""
    return nn.BCEWithLogitsLoss()

class ListwiseInfoNCELoss(nn.Module):
    """
    Listwise InfoNCE Contrastive Loss.
    Contrasts positive document vector against hard negatives and in-batch negatives.
    """
    def __init__(self, temperature: float = 0.07):
        super(ListwiseInfoNCELoss, self).__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, q_vecs: torch.Tensor, pos_doc_vecs: torch.Tensor, hard_neg_doc_vecs: torch.Tensor) -> torch.Tensor:
        batch_size, dim = q_vecs.shape
        K = hard_neg_doc_vecs.shape[1]
        
        pos_sim = torch.sum(q_vecs * pos_doc_vecs, dim=-1, keepdim=True) / self.temperature
        hard_neg_sim = torch.sum(q_vecs.unsqueeze(1) * hard_neg_doc_vecs, dim=-1) / self.temperature
        all_pos_sim = torch.matmul(q_vecs, pos_doc_vecs.T) / self.temperature
        
        mask = torch.eye(batch_size, device=q_vecs.device, dtype=torch.bool)
        in_batch_neg_sim = all_pos_sim.masked_select(~mask).view(batch_size, batch_size - 1)
        
        logits = torch.cat([pos_sim, hard_neg_sim, in_batch_neg_sim], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=q_vecs.device)
        
        loss = self.cross_entropy(logits, labels)
        return loss

