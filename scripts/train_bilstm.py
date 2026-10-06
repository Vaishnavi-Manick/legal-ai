import os
import sys
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root directory is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import AILADataLoader
from src.preprocessing import Vocabulary, clean_text
from src.dataset import BiLSTMPairDataset, collate_bilstm_pairs, pad_sequence
from src.models.bilstm import SiameseBiLSTMRelevanceModel
from src.loss import get_relevance_loss
from src.evaluation import calculate_metrics, save_loss_plot, save_metrics_csv, save_predictions_csv, save_history_csv


def main():
    print("=" * 60)
    print("      AILA 2019 BiLSTM RELEVANCE MODEL TRAINING")
    print("=" * 60)

    # 1. Hyperparameters & Configuration
    CONFIG = {
        "BATCH_SIZE": 32,
        "LEARNING_RATE": 0.001,
        "EPOCHS": 50,
        "EMBEDDING_DIM": 128,
        "HIDDEN_DIM": 128,
        "NUM_LAYERS": 1,
        "DROPOUT": 0.2,
        "MAX_QUERY_LENGTH": 256,
        "MAX_DOCUMENT_LENGTH": 512,
        "NUM_NEGATIVES": 5,
        "EARLY_STOPPING_PATIENCE": 10
    }

    # 2. Device Detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using compute device: {device}")
    if torch.cuda.is_available():
        print(f"[Device] GPU Model: {torch.cuda.get_device_name(0)}")

    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 3. Load AILA 2019 Data
    data_dir = os.path.join(PROJECT_ROOT, "data", "raw", "AILA2019")
    loader = AILADataLoader(data_dir)
    
    print("[Data] Loading queries, case documents, and relevance judgments...")
    queries = loader.load_queries()
    casedocs = loader.load_case_docs()
    qrels = loader.load_qrels_priorcases()

    train_qids, val_qids, test_qids = loader.get_query_splits(queries, qrels)
    print(f"[Data Split] Train Queries: {len(train_qids)} | Val Queries: {len(val_qids)} | Test Queries: {len(test_qids)}")

    # 4. Vocabulary Construction (No Data Leakage - fit only on Train Split)
    print("[Vocab] Fitting vocabulary strictly on training queries and training documents...")
    train_texts = [queries[q] for q in train_qids if q in queries]
    for qid in train_qids:
        for doc_id in qrels.get(qid, set()):
            if doc_id in casedocs:
                train_texts.append(casedocs[doc_id])

    vocab = Vocabulary()
    vocab.build_vocab(train_texts, max_vocab_size=30000, min_freq=2)
    print(f"[Vocab] Vocabulary built successfully. Size: {len(vocab)} words.")

    # 5. Dataset & DataLoaders
    print("[Dataset] Building training binary relevance pair dataset...")
    train_dataset = BiLSTMPairDataset(
        qids=train_qids,
        queries=queries,
        casedocs=casedocs,
        qrels=qrels,
        vocab=vocab,
        num_negatives=CONFIG["NUM_NEGATIVES"],
        max_q_len=CONFIG["MAX_QUERY_LENGTH"],
        max_d_len=CONFIG["MAX_DOCUMENT_LENGTH"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=True,
        collate_fn=lambda b: collate_bilstm_pairs(b, max_q_len=CONFIG["MAX_QUERY_LENGTH"], max_d_len=CONFIG["MAX_DOCUMENT_LENGTH"])
    )

    # 6. Instantiate Model, Loss Function, Optimizer
    model = SiameseBiLSTMRelevanceModel(
        vocab_size=len(vocab),
        embedding_dim=CONFIG["EMBEDDING_DIM"],
        hidden_dim=CONFIG["HIDDEN_DIM"],
        num_layers=CONFIG["NUM_LAYERS"],
        dropout=CONFIG["DROPOUT"]
    ).to(device)

    criterion = get_relevance_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["LEARNING_RATE"])

    # 7. SANITY CHECK (Before Full Training)
    print("\n" + "=" * 60)
    print("                SANITY CHECK ON 1 SMALL BATCH")
    print("=" * 60)
    sanity_batch = next(iter(train_loader))
    s_q_ids = sanity_batch["query_ids"].to(device)
    s_d_ids = sanity_batch["doc_ids"].to(device)
    s_labels = sanity_batch["labels"].to(device)

    model.eval()
    with torch.no_grad():
        s_logits = model(s_q_ids, s_d_ids)

    print(f"query tensor shape:    {s_q_ids.shape}")
    print(f"document tensor shape: {s_d_ids.shape}")
    print(f"label tensor shape:    {s_labels.shape}")
    print(f"model output shape:    {s_logits.shape}")

    # Sanity Backward Pass Check
    model.train()
    optimizer.zero_grad()
    s_out = model(s_q_ids, s_d_ids)
    s_loss = criterion(s_out, s_labels)
    s_loss.backward()
    optimizer.step()
    print(f"Sanity Loss Calculation: {s_loss.item():.4f}")
    print("Sanity Check Passed Successfully! Proceeding to Full Training...\n")

    # 8. Full Training & Validation Loop across 50 Epochs
    best_val_map = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []
    checkpoint_dir = os.path.join(PROJECT_ROOT, "models", "bilstm")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "bilstm_model.pt")

    # Build TFIDF candidate pools for fast validation and test scoring
    print("[Retrieval] Pre-computing candidate pools for fast evaluation...")
    from src.preprocessing import TFIDFRetriever
    tfidf_retriever = TFIDFRetriever(casedocs)
    bm25_candidate_pools = {}
    for qid in queries:
        ranked_docs = tfidf_retriever.rank_documents(queries[qid])
        top_docs = [doc_id for doc_id, _ in ranked_docs]
        pos_docs = list(qrels.get(qid, set()))
        combined = list(dict.fromkeys(pos_docs + top_docs[:300]))
        bm25_candidate_pools[qid] = combined

    def evaluate_model(eval_qids, candidate_pool_size=300):
        model.eval()
        rankings = {}
        
        with torch.no_grad():
            for qid in eval_qids:
                q_text = queries[qid]
                q_padded = pad_sequence(vocab.text_to_ids(q_text, max_len=CONFIG["MAX_QUERY_LENGTH"]), CONFIG["MAX_QUERY_LENGTH"])
                q_ids = torch.tensor([q_padded], dtype=torch.long, device=device)
                
                if candidate_pool_size is not None and qid in bm25_candidate_pools:
                    target_doc_ids = bm25_candidate_pools[qid][:candidate_pool_size]
                else:
                    target_doc_ids = list(casedocs.keys())
                    
                scores = []
                chunk_size = 512
                for i in range(0, len(target_doc_ids), chunk_size):
                    chunk_doc_ids = target_doc_ids[i:i+chunk_size]
                    doc_padded_list = [
                        pad_sequence(vocab.text_to_ids(casedocs[doc_id], max_len=CONFIG["MAX_DOCUMENT_LENGTH"]), CONFIG["MAX_DOCUMENT_LENGTH"])
                        for doc_id in chunk_doc_ids
                    ]
                    doc_tensor = torch.tensor(doc_padded_list, dtype=torch.long, device=device)
                    q_tensor = q_ids.repeat(len(chunk_doc_ids), 1)
                    
                    logits = model(q_tensor, doc_tensor)
                    probs = torch.sigmoid(logits).cpu().numpy().tolist()
                    scores.extend(probs)
                    
                doc_score_pairs = list(zip(target_doc_ids, scores))
                doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
                rankings[qid] = doc_score_pairs
                
        metrics = calculate_metrics(rankings, qrels, k_list=[5, 10])
        return metrics, rankings

    print("=" * 75)
    print(f"{'Epoch':<6} | {'Train Loss':<12} | {'Val Loss':<10} | {'Val MRR':<10} | {'Val MAP':<10} | {'Val NDCG@5':<10}")
    print("=" * 75)

    for epoch in range(1, CONFIG["EPOCHS"] + 1):
        model.train()
        total_train_loss = 0.0
        
        for batch in train_loader:
            q_ids = batch["query_ids"].to(device)
            d_ids = batch["doc_ids"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            logits = model(q_ids, d_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item() * len(labels)
            
        avg_train_loss = total_train_loss / len(train_dataset)

        # Validation Evaluation
        val_metrics, val_rankings = evaluate_model(val_qids, candidate_pool_size=150)
        val_map = val_metrics.get("MAP", 0.0)
        val_mrr = val_metrics.get("MRR", 0.0)
        val_ndcg5 = val_metrics.get("nDCG@5", 0.0)
        
        model.eval()
        val_bce_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for qid in val_qids:
                pos_docs = qrels.get(qid, set())
                non_relevant = list(set(casedocs.keys()) - pos_docs)[:10]
                eval_docs = list(pos_docs) + non_relevant
                for d_id in eval_docs:
                    if d_id in casedocs:
                        q_seq = pad_sequence(vocab.text_to_ids(queries[qid], max_len=CONFIG["MAX_QUERY_LENGTH"]), CONFIG["MAX_QUERY_LENGTH"])
                        d_seq = pad_sequence(vocab.text_to_ids(casedocs[d_id], max_len=CONFIG["MAX_DOCUMENT_LENGTH"]), CONFIG["MAX_DOCUMENT_LENGTH"])
                        q_tensor = torch.tensor([q_seq], dtype=torch.long, device=device)
                        d_tensor = torch.tensor([d_seq], dtype=torch.long, device=device)
                        target = torch.tensor([1.0 if d_id in pos_docs else 0.0], dtype=torch.float32, device=device)
                        logit = model(q_tensor, d_tensor)
                        l = criterion(logit, target)
                        val_bce_loss += l.item()
                        val_count += 1

        avg_val_loss = val_bce_loss / max(val_count, 1)

        history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_mrr": round(val_mrr, 4),
            "val_map": round(val_map, 4),
            "val_ndcg5": round(val_ndcg5, 4)
        })

        print(f"{epoch:<6} | {avg_train_loss:<12.4f} | {avg_val_loss:<10.4f} | {val_mrr:<10.4f} | {val_map:<10.4f} | {val_ndcg5:<10.4f}")

        if val_map > best_val_map:
            best_val_map = val_map
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "vocab": vocab.w2i,
                "config": CONFIG
            }, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["EARLY_STOPPING_PATIENCE"]:
                print(f"\n[Early Stopping] Triggered at epoch {epoch}. Best Val MAP: {best_val_map:.4f} at epoch {best_epoch}.")
                break

    print("=" * 75)
    print(f"Training Complete. Best Checkpoint saved to '{checkpoint_path}' (Epoch {best_epoch}, Val MAP: {best_val_map:.4f})")

    # 9. Load Best Checkpoint & Final Test Evaluation
    print("\n" + "=" * 60)
    print("             FINAL EVALUATION ON TEST SET (Q11-Q50)")
    print("=" * 60)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    test_metrics, test_rankings = evaluate_model(test_qids, candidate_pool_size=300)

    # 10. Save Artifacts
    results_dir = os.path.join(PROJECT_ROOT, "results", "bilstm")
    os.makedirs(results_dir, exist_ok=True)

    save_metrics_csv(test_metrics, model_name="BiLSTM", save_path=os.path.join(results_dir, "metrics.csv"))
    save_predictions_csv(test_rankings, qrels, save_path=os.path.join(results_dir, "predictions.csv"))
    save_history_csv(history, save_path=os.path.join(results_dir, "training_history.csv"))
    save_loss_plot(history, save_path=os.path.join(results_dir, "loss_plot.png"))

    print("\n" + "=" * 60)
    print("                   FINAL RESEARCH RESULTS")
    print("=" * 60)
    print(f"Model Name:  BiLSTM")
    print(f"P@5:        {test_metrics.get('P@5', 0.0):.4f}")
    print(f"Recall@5:   {test_metrics.get('Recall@5', 0.0):.4f}")
    print(f"MRR:        {test_metrics.get('MRR', 0.0):.4f}")
    print(f"MAP:        {test_metrics.get('MAP', 0.0):.4f}")
    print(f"NDCG@5:     {test_metrics.get('nDCG@5', 0.0):.4f}")
    print(f"NDCG@10:    {test_metrics.get('nDCG@10', 0.0):.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
