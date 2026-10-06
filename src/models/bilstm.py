import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AttentionPooling(nn.Module):
    """Additive attention pooling layer over sequence time steps."""
    def __init__(self, hidden_dim: int):
        super(AttentionPooling, self).__init__()
        self.attn_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_vector = nn.Parameter(torch.randn(hidden_dim, 1))
        nn.init.xavier_uniform_(self.attn_vector)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # hidden_states: (batch_size, seq_len, hidden_dim)
        # mask: (batch_size, seq_len) boolean tensor where True indicates valid token
        u = torch.tanh(self.attn_proj(hidden_states)) # (batch_size, seq_len, hidden_dim)
        attn_scores = torch.matmul(u, self.attn_vector).squeeze(-1) # (batch_size, seq_len)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, -1e9)
            
        attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1) # (batch_size, seq_len, 1)
        context = torch.sum(hidden_states * attn_weights, dim=1) # (batch_size, hidden_dim)
        return context

class SiameseBiLSTMRelevanceModel(nn.Module):
    """
    Siamese-style BiLSTM relevance model.
    Encodes query and document into dense representations u and v.
    Creates matching features:
    - query_representation (u)
    - document_representation (v)
    - absolute_difference (|u - v|)
    - elementwise_product (u * v)
    Concatenates matching features [u, v, |u - v|, u * v] and passes through MLP classifier.
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        pretrained_embeddings: np.ndarray = None
    ):
        super(SiameseBiLSTMRelevanceModel, self).__init__()
        
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                torch.tensor(pretrained_embeddings, dtype=torch.float32),
                freeze=False,
                padding_idx=0
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            nn.init.xavier_uniform_(self.embedding.weight)
            with torch.no_grad():
                self.embedding.weight[0] = 0.0

        self.dropout = nn.Dropout(p=dropout)
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        rep_dim = 2 * hidden_dim
        self.attention = AttentionPooling(hidden_dim=rep_dim)
        
        # Matching layer concatenation: [u, v, |u - v|, u * v] -> size 4 * rep_dim
        matching_dim = 4 * rep_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(matching_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1)
        )

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = (input_ids != 0)
        embeds = self.dropout(self.embedding(input_ids))
        lstm_out, _ = self.bilstm(embeds) # (batch, seq_len, 2 * hidden_dim)
        pooled = self.attention(lstm_out, mask=mask) # (batch, 2 * hidden_dim)
        return pooled

    def forward(self, query_ids: torch.Tensor, doc_ids: torch.Tensor) -> torch.Tensor:
        u = self.encode(query_ids) # query_representation (batch, rep_dim)
        v = self.encode(doc_ids)   # document_representation (batch, rep_dim)
        
        abs_diff = torch.abs(u - v)  # absolute_difference
        elem_prod = u * v            # elementwise_product
        
        matching = torch.cat([u, v, abs_diff, elem_prod], dim=-1) # (batch, 4 * rep_dim)
        logits = self.mlp(matching).squeeze(-1) # (batch,)
        return logits

