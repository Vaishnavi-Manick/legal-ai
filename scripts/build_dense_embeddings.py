import os
import sys
import json
import torch
import torch.nn.functional as F
from datetime import datetime
from transformers import AutoTokenizer, AutoModel

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import AILADataLoader


def mean_pooling(model_output, attention_mask):
    """Extracts normalized sentence/document embeddings from Transformer outputs."""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def main():
    print("=" * 70)
    print("      BUILDING DENSE DOCUMENT EMBEDDINGS INDEX (AILA 2019)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Compute device: {device}")

    # 1. Load AILA 2019 Corpus Documents
    data_dir = os.path.join(PROJECT_ROOT, "data", "raw", "AILA2019")
    loader = AILADataLoader(data_dir=data_dir)
    casedocs = loader.load_case_docs()
    doc_ids = list(casedocs.keys())
    num_docs = len(doc_ids)
    print(f"[Data] Loaded {num_docs} AILA case documents.")

    # 2. Load Pretrained Model & Tokenizer
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"[Model] Loading pretrained embedding model '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # Programmatically detect embedding dimension
    embedding_dim = getattr(model.config, "hidden_size", 384)
    print(f"[Model] Programmatically detected embedding dimension: {embedding_dim}")

    # 3. Generate Embeddings in Batches with L2 Normalization
    print(f"[Embedding] Encoding {num_docs} documents in batches...")
    embeddings_list = []
    batch_size = 32

    with torch.no_grad():
        for i in range(0, num_docs, batch_size):
            batch_doc_ids = doc_ids[i : i + batch_size]
            batch_texts = [casedocs[doc_id][:1500] for doc_id in batch_doc_ids]

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(device)

            outputs = model(**encoded)
            embeds = mean_pooling(outputs, encoded["attention_mask"])
            normalized_embeds = F.normalize(embeds, p=2, dim=1)
            embeddings_list.append(normalized_embeds.cpu())

    all_embeddings = torch.cat(embeddings_list, dim=0)
    print(f"[Embedding] Final embeddings matrix shape: {all_embeddings.shape}")

    # 4. Save Artifacts to models/dense_retrieval/
    output_dir = os.path.join(PROJECT_ROOT, "models", "dense_retrieval")
    os.makedirs(output_dir, exist_ok=True)

    embeddings_path = os.path.join(output_dir, "document_embeddings.pt")
    metadata_path = os.path.join(output_dir, "metadata.json")

    torch.save(all_embeddings, embeddings_path)
    print(f"[Artifact] Saved document embeddings to '{embeddings_path}'")

    metadata = {
        "model_name": model_name,
        "embedding_dimension": int(all_embeddings.shape[1]),
        "number_of_documents": num_docs,
        "normalization_method": "L2",
        "dataset_name": "AILA 2019",
        "generation_date": datetime.now().isoformat(),
        "case_ids": doc_ids
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[Artifact] Saved dense retrieval metadata to '{metadata_path}'")
    print("=" * 70)
    print("   DENSE DOCUMENT EMBEDDINGS BUILD COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
