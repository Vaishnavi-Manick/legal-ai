import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig

class LegalTransformerCrossEncoder(nn.Module):
    """
    Cross-Encoder Legal Transformer Relevance Model.
    Encodes [CLS] Query [SEP] Document Chunk [SEP] and aggregates chunk relevance scores
    using Max-Pooling (MaxP) over chunks to handle long legal case documents.
    """
    def __init__(self, model_name: str = "nlpaueb/legal-bert-base-uncased", dropout: float = 0.1):
        super().__init__()
        self.model_name = model_name
        self.config = AutoConfig.from_pretrained(model_name, num_labels=1)
        self.config.hidden_dropout_prob = dropout
        self.config.attention_probs_dropout_prob = dropout
        
        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=self.config
        )
        
    def forward(self, input_ids, attention_mask, token_type_ids=None, chunk_counts=None):
        """
        Forward pass for query-document chunk pairs.
        
        input_ids: [total_chunks, seq_len]
        attention_mask: [total_chunks, seq_len]
        chunk_counts: list of integers indicating how many chunks belong to each document sample.
        """
        if token_type_ids is not None:
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        else:
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
        flat_logits = outputs.logits.squeeze(-1) # [total_chunks]
        
        if chunk_counts is None:
            return flat_logits
            
        # Max-Pooling Aggregation across chunks for each document sample in batch
        doc_logits = []
        offset = 0
        for count in chunk_counts:
            chunk_logits = flat_logits[offset : offset + count]
            max_logit = torch.max(chunk_logits)
            doc_logits.append(max_logit)
            offset += count
            
        return torch.stack(doc_logits)
