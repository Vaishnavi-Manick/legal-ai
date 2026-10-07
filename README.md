# Legal AI: AILA 2019 Legal Information Retrieval & Precedent Ranking

Supervised legal query-document relevance ranking and precedent retrieval system trained on the **AILA 2019 dataset**.

## 📁 Project Structure

```
legal_ai/
│
├── data/
│   ├── raw/
│   │   ├── AILA2019/                # Raw AILA 2019 dataset (queries, casedocs, qrels)
│   │   └── data.zip                 # Dataset ZIP archive
│   └── processed/                   # Processed cached data / features
│
├── notebooks/
│   ├── 01_data_preparation.ipynb    # Dataset loading & corpus stats
│   ├── 02_bilstm.ipynb              # BiLSTM model training & evaluation notebook
│   ├── 03_transformer.ipynb         # Legal Transformer Cross-Encoder notebook
│   └── 05_model_comparison.ipynb   # Comparative TREC metrics analysis
│
├── src/
│   ├── data_loader.py               # Dataset loading & train/val/test query splitting
│   ├── preprocessing.py            # Text cleaning, vocabulary, TF-IDF retriever
│   ├── dataset.py                   # BiLSTM pair dataset & collate functions
│   ├── transformer_dataset.py       # Sliding-window chunking & transformer dataset
│   ├── loss.py                      # BCE & InfoNCE loss functions
│   ├── utils.py                     # TREC IR metrics calculation & plotting helpers
│   ├── training.py                  # Seed setting & device management utilities
│   ├── evaluation.py                # Reusable evaluation API wrapper
│   ├── inference.py                 # LegalSearchEngine two-stage inference engine
│   └── models/
│       ├── bilstm.py                # Siamese BiLSTM Dual-Encoder architecture
│       └── transformer.py           # Legal Transformer Cross-Encoder architecture
│
├── models/
│   ├── bilstm/
│   │   └── bilstm_model.pt          # Trained BiLSTM PyTorch model checkpoint
│   └── transformer/
│       ├── transformer_model.pt     # Trained Transformer model checkpoint
│       ├── best_checkpoint.pt       # Checkpoint weights
│       ├── config.json              # Transformer model configuration
│       ├── model.safetensors        # Safetensors model weights
│       └── tokenizer.json           # Tokenizer definition
│
├── results/
│   ├── bilstm/
│   │   ├── metrics.csv              # BiLSTM test set metrics
│   │   ├── predictions.csv          # Detailed test set predictions
│   │   ├── training_history.csv     # Training & validation history across epochs
│   │   └── loss_plot.png            # Training vs Validation loss curve plot
│   │
│   ├── transformer/
│   │   ├── metrics.csv              # Transformer test set metrics
│   │   ├── predictions.csv          # Detailed test set predictions
│   │   ├── training_history.csv     # Training & validation history across epochs
│   │   └── loss_plot.png            # Training vs Validation loss curve plot
│   │
│   └── comparison/
│       ├── model_comparison.csv     # Evaluated model performance comparison
│       └── README.md                # Research evaluation & model selection report
│
├── frontend/
│   └── app.py                       # Streamlit Web UI application
│
├── research/
│   └── papers/                      # Research papers on Legal Information Retrieval & RAG
│
├── scripts/
│   ├── train_bilstm.py              # Script to train & evaluate BiLSTM model
│   └── train_transformer.py         # Script to train & evaluate Transformer model
│
├── requirements.txt                 # Project dependencies
└── README.md                        # Project documentation
```

## 📊 Final Comparative Evaluation Results

| Model Architecture | Precision@5 | Recall@5 | MRR | MAP | NDCG@5 | NDCG@10 | Selected |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Siamese BiLSTM + Attention** | 0.0000 | 0.0000 | 0.0082 | 0.0068 | 0.0000 | 0.0051 | ❌ |
| **Legal Transformer Cross-Encoder** | **0.1250** | **0.1881** | **0.2790** | **0.1886** | **0.1774** | **0.2008** | **✅ Final Selected Model** |

---

## 🚀 Execution Instructions

### Running Streamlit Web UI
```bash
python -m streamlit run frontend/app.py
```

### Running Model Training Scripts
```bash
# Train BiLSTM Model
python scripts/train_bilstm.py

# Train Legal Transformer Cross-Encoder
python scripts/train_transformer.py
```
