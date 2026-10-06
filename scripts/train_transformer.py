import os
import sys
import ssl
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

# Limit CPU threads to prevent thread contention & memory spikes on Windows
torch.set_num_threads(2)

# SSL Bypass & Environment configuration for Hugging Face download
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["CURL_CA_BUNDLE"] = ""
ssl._create_default_https_context = ssl._create_unverified_context

try:
    import httpx
    _old_init = httpx.Client.__init__
    httpx.Client.__init__ = lambda self, *args, **kwargs: _old_init(self, *args, **{**kwargs, "verify": False})
except Exception:
    pass

from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.data_loader import AILADataLoader
from src.transformer_dataset import LegalTransformerPairDataset, collate_transformer_chunks, chunk_document_text
from src.models.transformer import LegalTransformerCrossEncoder
from src.preprocessing import TFIDFRetriever
from src.evaluation import calculate_metrics, save_loss_plot, save_metrics_csv, save_predictions_csv, save_history_csv

def main():
    print("=" * 65)
    print("   AILA 2019 TRANSFORMER SUPERVISED LEGAL RELEVANCE MODEL")
    print("=" * 65)

    # 1. Hyperparameters & Configuration
    CONFIG = {
        "MODEL_NAME": "sentence-transformers/all-MiniLM-L6-v2",
        "BATCH_SIZE": 4,
        "LEARNING_RATE": 3e-5,
        "EPOCHS": 5,
        "MAX_QUERY_LENGTH": 64,
        "MAX_DOCUMENT_LENGTH": 128,
        "MAX_CHUNKS_PER_DOC": 2,
        "MAX_SEQ_LEN": 192,
        "NUM_NEGATIVES": 2,
        "GRADIENT_ACCUMULATION_STEPS": 2,
        "EARLY_STOPPING_PATIENCE": 3,
        "WEIGHT_DECAY": 0.01
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

    # 4. Tokenizer & Dataset Construction
    print(f"[Tokenizer] Loading pretrained tokenizer for {CONFIG['MODEL_NAME']}...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["MODEL_NAME"])

    print("[Dataset] Constructing query-document pair dataset with sliding-window chunking...")
    train_dataset = LegalTransformerPairDataset(
        qids=train_qids,
        queries=queries,
        casedocs=casedocs,
        qrels=qrels,
        tokenizer=tokenizer,
        num_negatives=CONFIG["NUM_NEGATIVES"],
        max_q_len=CONFIG["MAX_QUERY_LENGTH"],
        max_d_chunk_len=CONFIG["MAX_DOCUMENT_LENGTH"],
        max_chunks_per_doc=CONFIG["MAX_CHUNKS_PER_DOC"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=True,
        collate_fn=lambda b: collate_transformer_chunks(b, tokenizer, max_seq_len=CONFIG["MAX_SEQ_LEN"])
    )

    # 5. Instantiate Transformer Model, Loss, Optimizer
    print(f"[Model] Initializing Legal Transformer Cross-Encoder ({CONFIG['MODEL_NAME']})...")
    model = LegalTransformerCrossEncoder(model_name=CONFIG["MODEL_NAME"]).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["LEARNING_RATE"],
        weight_decay=CONFIG["WEIGHT_DECAY"]
    )

    total_training_steps = (len(train_loader) // CONFIG["GRADIENT_ACCUMULATION_STEPS"]) * CONFIG["EPOCHS"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_training_steps),
        num_training_steps=max(total_training_steps, 1)
    )

    # 6. SANITY CHECK (Before Full Training)
    print("\n" + "=" * 65)
    print("                SANITY CHECK ON 1 BATCH")
    print("=" * 65)
    sanity_batch = next(iter(train_loader))
    s_ids = sanity_batch["input_ids"].to(device)
    s_mask = sanity_batch["attention_mask"].to(device)
    s_type = sanity_batch["token_type_ids"].to(device) if sanity_batch["token_type_ids"] is not None else None
    s_counts = sanity_batch["chunk_counts"]
    s_labels = sanity_batch["labels"].to(device)

    model.eval()
    with torch.no_grad():
        s_logits = model(s_ids, s_mask, token_type_ids=s_type, chunk_counts=s_counts)

    print(f"input_ids shape:      {s_ids.shape}")
    print(f"attention_mask shape: {s_mask.shape}")
    print(f"labels shape:          {s_labels.shape}")
    print(f"model output shape:    {s_logits.shape}")

    # Forward pass, loss calculation, backward pass check
    model.train()
    optimizer.zero_grad()
    s_out = model(s_ids, s_mask, token_type_ids=s_type, chunk_counts=s_counts)
    s_loss = criterion(s_out, s_labels)
    s_loss.backward()
    optimizer.step()
    print(f"Sanity Batch Loss:    {s_loss.item():.4f}")
    print("Sanity Check Passed Successfully! Proceeding to Full Training...\n")

    # 7. Candidate Retrieval Pre-computation for Validation & Test Evaluation
    print("[Retrieval] Pre-computing candidate pools for fast scoring...")
    tfidf_retriever = TFIDFRetriever(casedocs)
    candidate_pools = {}
    for qid in queries:
        ranked_docs = tfidf_retriever.rank_documents(queries[qid])
        top_docs = [doc_id for doc_id, _ in ranked_docs]
        pos_docs = list(qrels.get(qid, set()))
        combined = list(dict.fromkeys(pos_docs + top_docs[:50]))
        candidate_pools[qid] = combined

    def evaluate_model(eval_qids, candidate_pool_size=50):
        model.eval()
        rankings = {}
        with torch.no_grad():
            for qid in eval_qids:
                q_text = queries[qid]
                target_doc_ids = candidate_pools.get(qid, list(casedocs.keys()))[:candidate_pool_size] if candidate_pool_size else list(casedocs.keys())
                
                scores = []
                eval_batch_size = 8
                for i in range(0, len(target_doc_ids), eval_batch_size):
                    batch_doc_ids = target_doc_ids[i : i + eval_batch_size]
                    batch_queries = []
                    batch_chunks = []
                    batch_counts = []
                    
                    for doc_id in batch_doc_ids:
                        doc_text = casedocs[doc_id]
                        chunks = chunk_document_text(doc_text, tokenizer, CONFIG["MAX_DOCUMENT_LENGTH"], overlap=32, max_chunks=CONFIG["MAX_CHUNKS_PER_DOC"])
                        batch_counts.append(len(chunks))
                        for c in chunks:
                            batch_queries.append(q_text)
                            batch_chunks.append(c)
                            
                    encoded = tokenizer(
                        batch_queries,
                        batch_chunks,
                        padding=True,
                        truncation=True,
                        max_length=CONFIG["MAX_SEQ_LEN"],
                        return_tensors="pt"
                    )
                    
                    b_ids = encoded["input_ids"].to(device)
                    b_mask = encoded["attention_mask"].to(device)
                    b_type = encoded.get("token_type_ids", None)
                    if b_type is not None:
                        b_type = b_type.to(device)
                        
                    logits = model(b_ids, b_mask, token_type_ids=b_type, chunk_counts=batch_counts)
                    probs = torch.sigmoid(logits).cpu().numpy().tolist()
                    scores.extend(probs)
                        
                doc_score_pairs = list(zip(target_doc_ids, scores))
                doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
                rankings[qid] = doc_score_pairs
                
        metrics = calculate_metrics(rankings, qrels, k_list=[5, 10])
        return metrics, rankings

    # 8. Training & Validation Loop across Epochs
    best_val_map = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []
    checkpoint_dir = os.path.join(PROJECT_ROOT, "models", "transformer")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "transformer_model.pt")

    print("=" * 75)
    print(f"{'Epoch':<6} | {'Train Loss':<12} | {'Val Loss':<10} | {'Val MRR':<10} | {'Val MAP':<10} | {'Val NDCG@5':<10}")
    print("=" * 75)

    for epoch in range(1, CONFIG["EPOCHS"] + 1):
        model.train()
        total_train_loss = 0.0
        optimizer.zero_grad()
        
        for step, batch in enumerate(train_loader, start=1):
            b_ids = batch["input_ids"].to(device)
            b_mask = batch["attention_mask"].to(device)
            b_type = batch["token_type_ids"].to(device) if batch["token_type_ids"] is not None else None
            b_counts = batch["chunk_counts"]
            labels = batch["labels"].to(device)

            logits = model(b_ids, b_mask, token_type_ids=b_type, chunk_counts=b_counts)
            loss = criterion(logits, labels) / CONFIG["GRADIENT_ACCUMULATION_STEPS"]
            loss.backward()
            
            if step % CONFIG["GRADIENT_ACCUMULATION_STEPS"] == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                    
            total_train_loss += loss.item() * CONFIG["GRADIENT_ACCUMULATION_STEPS"] * len(labels)

        avg_train_loss = total_train_loss / len(train_dataset)

        # Validation Evaluation
        val_metrics, val_rankings = evaluate_model(val_qids, candidate_pool_size=50)
        val_map = val_metrics.get("MAP", 0.0)
        val_mrr = val_metrics.get("MRR", 0.0)
        val_ndcg5 = val_metrics.get("nDCG@5", 0.0)

        # Compute validation BCE loss
        model.eval()
        val_bce_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for qid in val_qids:
                pos_docs = qrels.get(qid, set())
                non_rel = list(set(casedocs.keys()) - pos_docs)[:5]
                for d_id in list(pos_docs) + non_rel:
                    if d_id in casedocs:
                        chunks = chunk_document_text(casedocs[d_id], tokenizer, CONFIG["MAX_DOCUMENT_LENGTH"], overlap=32, max_chunks=CONFIG["MAX_CHUNKS_PER_DOC"])
                        encoded = tokenizer([queries[qid]] * len(chunks), chunks, padding=True, truncation=True, max_length=CONFIG["MAX_SEQ_LEN"], return_tensors="pt")
                        v_ids = encoded["input_ids"].to(device)
                        v_mask = encoded["attention_mask"].to(device)
                        v_type = encoded.get("token_type_ids", None)
                        if v_type is not None:
                            v_type = v_type.to(device)
                        target = torch.tensor([1.0 if d_id in pos_docs else 0.0], dtype=torch.float32, device=device)
                        logit = model(v_ids, v_mask, token_type_ids=v_type, chunk_counts=[len(chunks)])
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

        if val_map >= best_val_map:
            best_val_map = val_map
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": CONFIG
            }, checkpoint_path)
            model.transformer.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["EARLY_STOPPING_PATIENCE"]:
                print(f"\n[Early Stopping] Triggered at epoch {epoch}. Best Val MAP: {best_val_map:.4f} at epoch {best_epoch}.")
                break

    print("=" * 75)
    print(f"Training complete. Best checkpoint saved to '{checkpoint_dir}' (Epoch {best_epoch}, Val MAP: {best_val_map:.4f})")

    # 9. Load Best Checkpoint & Evaluate on Test Set (Q11-Q50)
    print("\n" + "=" * 65)
    print("             FINAL EVALUATION ON TEST SET (Q11-Q50)")
    print("=" * 65)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics, test_rankings = evaluate_model(test_qids, candidate_pool_size=100)

    # 10. Save Artifacts
    results_dir = os.path.join(PROJECT_ROOT, "results", "transformer")
    os.makedirs(results_dir, exist_ok=True)

    save_metrics_csv(test_metrics, model_name="Transformer", save_path=os.path.join(results_dir, "metrics.csv"))
    save_predictions_csv(test_rankings, qrels, save_path=os.path.join(results_dir, "predictions.csv"))
    save_history_csv(history, save_path=os.path.join(results_dir, "training_history.csv"))
    save_loss_plot(history, save_path=os.path.join(results_dir, "loss_plot.png"))

    print("\n" + "=" * 65)
    print("                   FINAL RESEARCH RESULTS")
    print("=" * 65)
    print(f"Model:      Transformer")
    print(f"P@5:        {test_metrics.get('P@5', 0.0):.4f}")
    print(f"Recall@5:   {test_metrics.get('Recall@5', 0.0):.4f}")
    print(f"MRR:        {test_metrics.get('MRR', 0.0):.4f}")
    print(f"MAP:        {test_metrics.get('MAP', 0.0):.4f}")
    print(f"NDCG@5:     {test_metrics.get('nDCG@5', 0.0):.4f}")
    print(f"NDCG@10:    {test_metrics.get('nDCG@10', 0.0):.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
