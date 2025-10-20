[English](./README.md) | [简体中文](./README_zh-CN.md)

# 🔍 ICLR 2026 Paper Search

Search through 18,000+ ICLR 2026 submissions using semantic similarity.

## What is This?

A simple search tool to find research papers by describing what you're looking for in natural language. Just type your query and get relevant papers instantly!

## Features

- 🔎 **Natural Language Search** - Describe papers in plain English
- ⚡ **Instant Results** - Pre-computed embeddings for fast search
- 🎯 **Smart Filtering** - Filter by research area
- 📊 **18,000+ Papers** - All ICLR 2026 submissions
- 🆓 **Free & Open Source**

## How to Use

1. **Enter your search query** - Describe what papers you're looking for
2. **Adjust settings** (optional) - Number of results, research area filters
3. **Click Search** - Get papers ranked by relevance

### Example Queries

- "vision transformers for image classification"
- "efficient attention mechanisms for long sequences"
- "few-shot learning with meta-learning"
- "diffusion models for image generation"
- "graph neural networks for molecular property prediction"

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Visit `http://localhost:7860`

## Deploy to Hugging Face Spaces

1. Create a new Space at https://huggingface.co/spaces
2. Push your code:
   ```bash
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME
   git push hf main
   ```
3. Your search tool will be live!

**Note**: Make sure to upload `iclr2026_papers.json` and the cache file `output/cache_*.npy` to the Space.

## How It Works

Papers are converted to embeddings (numerical vectors) that capture their semantic meaning. When you search, your query is converted to the same format and we find papers with the most similar vectors using cosine similarity.

## Technology

- **Framework**: Gradio
- **Embedding Model**: all-MiniLM-L6-v2 (fast, 384 dimensions)
- **Dataset**: ICLR 2026 submissions from OpenReview

## Citation

```bibtex
@misc{SearchPaperByEmbedding,
  author = {gyj155},
  title = {ICLR 2026 Paper Search},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/gyj155/SearchPaperByEmbedding}}
}
```

## License

MIT License

---

⭐ If this helps you find papers, please star the repo!
