"""
Gradio app for Paper Semantic Search.
Pre-loaded with ICLR 2026 papers for instant searching.
Optimized for Hugging Face Spaces.
"""

import gradio as gr
import json
import os
import pandas as pd
from pathlib import Path
from src.search import PaperSearcher

# Configuration
PAPERS_FILE = "iclr2026_papers.json"
DEFAULT_MODEL_TYPE = "local"
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

# Global searcher instance
searcher = None
papers = []
primary_areas = []


def initialize_app():
    """Initialize the app by loading papers and computing embeddings."""
    global searcher, papers, primary_areas
    
    if not Path(PAPERS_FILE).exists():
        return f"❌ Papers file '{PAPERS_FILE}' not found. Please ensure the file exists."
    
    try:
        # Load papers
        with open(PAPERS_FILE, "r", encoding="utf-8") as f:
            papers_data = json.load(f)
        
        # Handle different JSON formats
        if isinstance(papers_data, dict) and "results" in papers_data:
            papers = [r["paper"] for r in papers_data["results"]]
        elif isinstance(papers_data, dict):
            papers = [papers_data]
        else:
            papers = papers_data
        
        # Extract primary areas
        primary_areas = sorted(list(set(p.get("primary_area", "N/A") for p in papers)))
        
        # Initialize searcher with default model
        searcher = PaperSearcher(
            PAPERS_FILE,
            model_type=DEFAULT_MODEL_TYPE,
            model_name=DEFAULT_MODEL_NAME,
        )
        
        # Compute embeddings (will use cache if available)
        searcher.compute_embeddings()
        
        return f"✅ Loaded {len(papers)} papers. Ready to search using {searcher.model.model_name}!"
        
    except Exception as e:
        return f"❌ Error initializing app: {str(e)}"


def search_papers(query, top_k, filter_areas):
    """Search for similar papers."""
    global searcher
    
    if searcher is None:
        return "❌ App not initialized. Please refresh the page.", None
    
    if not query or not query.strip():
        return "⚠️ Please enter a search query.", None
    
    try:
        # Perform search
        results = searcher.search(query=query.strip(), top_k=int(top_k))
        
        # Filter by primary areas if specified
        if filter_areas:
            results = [
                r for r in results
                if r["paper"].get("primary_area", "N/A") in filter_areas
            ]
        
        if not results:
            return "⚠️ No results found matching your criteria.", None
        
        # Format results for display
        results_data = []
        for i, result in enumerate(results, 1):
            paper = result["paper"]
            title = paper.get("title", "N/A")
            url = paper.get("forum_url", "N/A")
            
            # Create clickable link for title
            if url != "N/A":
                title_link = f'<a href="{url}" target="_blank">{title}</a>'
            else:
                title_link = title
            
            results_data.append({
                "Rank": i,
                "Score": f"{result['similarity']:.4f}",
                "Title": title,
                "Area": paper.get("primary_area", "N/A"),
                "Number": f"#{paper.get('number', 'N/A')}",
                "URL": url,
            })
        
        df = pd.DataFrame(results_data)
        
        success_msg = f"✅ Found {len(results)} relevant papers (similarity scores: {results[0]['similarity']:.4f} - {results[-1]['similarity']:.4f})"
        return success_msg, df
        
    except Exception as e:
        return f"❌ Search error: {str(e)}", None


def get_example_query(example_type):
    """Return example queries for different research areas."""
    examples = {
        "Computer Vision": "3D scene understanding and object detection using deep neural networks",
        "NLP": "Large language models for reasoning and instruction following",
        "Reinforcement Learning": "Policy optimization and model-based reinforcement learning for robotics",
        "Generative Models": "Diffusion models and variational autoencoders for image generation",
        "Theory": "Optimization theory and convergence analysis for neural networks",
        "ML Systems": "Efficient training and inference systems for large-scale machine learning",
    }
    return examples.get(example_type, "")


# Initialize the app
init_status = initialize_app()
print(init_status)

# Create Gradio interface
with gr.Blocks(title="ICLR 2026 Paper Search", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        f"""
        # 🔍 ICLR 2026 Paper Semantic Search
        
        Search through **{len(papers)} ICLR 2026 submissions** using semantic similarity.
        
        {init_status}
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 🔎 Search Query")
            
            query_input = gr.Textbox(
                label="What papers are you looking for?",
                placeholder="Describe the papers you're interested in...\n\nExamples:\n• 'vision-language models for embodied AI'\n• 'efficient transformers with linear attention'\n• 'few-shot learning with meta-learning'",
                lines=4,
            )
            
            with gr.Row():
                with gr.Column(scale=2):
                    example_type = gr.Dropdown(
                        choices=[
                            "Computer Vision",
                            "NLP",
                            "Reinforcement Learning",
                            "Generative Models",
                            "Theory",
                            "ML Systems",
                        ],
                        label="Quick Examples",
                        value="Computer Vision",
                    )
                with gr.Column(scale=1):
                    load_example_btn = gr.Button("📝 Load Example", size="sm")
            
            search_button = gr.Button("🚀 Search Papers", variant="primary", size="lg")
            
            status_output = gr.Textbox(
                label="Status",
                interactive=False,
                show_label=False,
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Search Settings")
            
            top_k = gr.Slider(
                minimum=5,
                maximum=200,
                value=50,
                step=5,
                label="Number of Results",
                info="How many papers to return",
            )
            
            filter_areas = gr.CheckboxGroup(
                choices=primary_areas,
                label="Filter by Research Area",
                info="Leave empty to search all areas",
            )
            
            gr.Markdown(
                f"""
                ### 📊 Dataset Info
                
                - **Conference**: ICLR 2026
                - **Papers**: {len(papers)}
                - **Model**: {DEFAULT_MODEL_NAME}
                - **Areas**: {len(primary_areas)}
                """
            )
    
    gr.Markdown("### 📊 Search Results")
    
    results_output = gr.Dataframe(
        headers=["Rank", "Score", "Title", "Area", "Number", "URL"],
        datatype=["number", "str", "str", "str", "str", "str"],
        wrap=True,
        interactive=False,
        height=500,
    )
    
    gr.Markdown(
        """
        ---
        ### 💡 Search Tips
        
        - **Be specific**: Describe the exact problem, method, or domain you're interested in
        - **Use technical terms**: The model understands research terminology well
        - **Combine concepts**: Try queries like "graph neural networks for drug discovery"
        - **Filter by area**: Narrow down results by selecting specific research areas
        
        ### 🎯 How it Works
        
        This tool uses semantic search powered by sentence embeddings. Your query is converted into a vector,
        and we find papers whose titles and abstracts have the most similar vectors using cosine similarity.
        
        ### 🔗 Links
        
        - [GitHub Repository](https://github.com/gyj155/SearchPaperByEmbedding)
        - [ICLR 2026](https://iclr.cc/Conferences/2026)
        """
    )
    
    # Event handlers
    load_example_btn.click(
        fn=get_example_query,
        inputs=[example_type],
        outputs=[query_input],
    )
    
    search_button.click(
        fn=search_papers,
        inputs=[query_input, top_k, filter_areas],
        outputs=[status_output, results_output],
    )
    
    # Also trigger search on Enter key
    query_input.submit(
        fn=search_papers,
        inputs=[query_input, top_k, filter_areas],
        outputs=[status_output, results_output],
    )


# For HuggingFace Spaces and local deployment
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        share=False,
        show_error=True,
    )
