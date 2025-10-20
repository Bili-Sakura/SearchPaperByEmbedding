"""
Script to embed all papers using Qwen3-Embedding-8B model.
This will load papers from iclr2026_papers.json and save embeddings to output directory.
Uses sentence-transformers best practices with correct last-token pooling.
Optimized for single 80GB GPU with float16 precision.
"""

import os
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Configuration
PAPERS_FILE = "path/to/iclr2026_papers.json"
MODEL_PATH = "path/to/Qwen/Qwen3-Embedding-8B"
OUTPUT_DIR = "path/to/output"
BATCH_SIZE = 256  # Optimized for 80GB GPU
MAX_LENGTH = 1024  # Maximum sequence length


def create_text_from_paper(paper):
    """Create text representation from paper for embedding."""
    parts = []
    if paper.get("title"):
        parts.append(f"Title: {paper['title']}")
    if paper.get("abstract"):
        parts.append(f"Abstract: {paper['abstract']}")
    if paper.get("keywords"):
        kw = (
            ", ".join(paper["keywords"])
            if isinstance(paper["keywords"], list)
            else paper["keywords"]
        )
        parts.append(f"Keywords: {kw}")
    return " ".join(parts)


def main():
    print("=" * 80)
    print("ICLR 2026 Papers Embedding with Qwen3-Embedding-8B")
    print("Using sentence-transformers best practices (single GPU)")
    print("=" * 80)
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load papers
    print(f"\n[1/4] Loading papers from: {PAPERS_FILE}")
    with open(PAPERS_FILE, "r", encoding="utf-8") as f:
        papers = json.load(f)
    print(f"✓ Loaded {len(papers)} papers")
    
    # Prepare texts for embedding
    print(f"\n[2/4] Preparing texts for embedding...")
    texts = [create_text_from_paper(paper) for paper in papers]
    print(f"✓ Prepared {len(texts)} text entries")
    
    # Load model with best practices
    print(f"\n[3/4] Loading model with optimal configuration...")
    print("This may take a few minutes...")
    
    model = SentenceTransformer(
        MODEL_PATH,
        device="cuda:0",  # Single GPU is sufficient (80GB)
        model_kwargs={
            "torch_dtype": "float16",  # FP16 for efficiency
            "trust_remote_code": True
        },
        tokenizer_kwargs={
            "padding_side": "left",  # Required for Qwen3
            "trust_remote_code": True
        },
        trust_remote_code=True
    )
    
    print(f"✓ Model loaded and optimized")
    print(f"  - Loaded on cuda:0 (80GB is sufficient)")
    print(f"  - Last token pooling (correct for Qwen3)")
    print(f"  - Using float16 for efficiency")
    
    # Compute embeddings
    print(f"\n[4/4] Computing embeddings...")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Total batches: {(len(texts) + BATCH_SIZE - 1) // BATCH_SIZE}")
    print(f"  - Max length: {MAX_LENGTH}")
    
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,  # Normalize for cosine similarity
        convert_to_numpy=True
    )
    
    print(f"✓ Embeddings computed: shape={embeddings.shape}, dtype={embeddings.dtype}")
    
    # Save embeddings
    output_file = Path(OUTPUT_DIR) / "embeddings_qwen3_8b.npy"
    print(f"\n[5/5] Saving embeddings to: {output_file}")
    np.save(output_file, embeddings)
    print(f"✓ Embeddings saved successfully")
    
    # Save metadata
    metadata = {
        "model_path": MODEL_PATH,
        "model_name": "Qwen3-Embedding-8B",
        "papers_file": PAPERS_FILE,
        "num_papers": len(papers),
        "embedding_dim": embeddings.shape[1],
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "method": "sentence-transformers",
        "pooling": "last_token",  # Correct pooling for Qwen3
        "device": "cuda:0",
        "dtype": "float16",
        "normalized": True
    }
    
    metadata_file = Path(OUTPUT_DIR) / "embeddings_qwen3_8b_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"✓ Metadata saved to: {metadata_file}")
    
    print("\n" + "=" * 80)
    print("EMBEDDING COMPLETE!")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  - Papers processed: {len(papers)}")
    print(f"  - Embedding dimension: {embeddings.shape[1]}")
    print(f"  - Method: sentence-transformers (best practice)")
    print(f"  - Pooling: last_token (correct for Qwen3)")
    print(f"  - Output file: {output_file}")
    print(f"  - File size: {output_file.stat().st_size / (1024**2):.2f} MB")
    print("\nYou can now use these embeddings for semantic search!")


if __name__ == "__main__":
    main()

