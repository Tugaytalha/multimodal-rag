import gradio as gr
import pandas as pd
import os
import time
from pathlib import Path
import numpy as np
from shutil import copy2
import logging

# Custom modules
try:
    from multimodal_rag_system import create_multimodal_rag_system, MultimodalRAGSystem
except ImportError:
    from multimodal_rag_system import create_multimodal_rag_system, MultimodalRAGSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration constants
TEXT_EMBEDDING_MODELS = [
    "jinaai/jina-embeddings-v4",
    "jinaai/jina-embeddings-v3",
    "intfloat/multilingual-e5-large-instruct",
    "Omerhan/intfloat-fine-tuned-14376-v4",
    "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
]

EMBEDDING_MODELS = TEXT_EMBEDDING_MODELS  # For backward compatibility

MULTIMODAL_EMBEDDING_MODELS = [
    "jinaai/jina-embeddings-v4"
]

# ------------------------
# 🔄 Fetch available models from Ollama
# ------------------------
try:
    import ollama

    _ollama_resp = ollama.list()
    _OLLAMA_ALL_MODELS = [m.get("model", "") for m in _ollama_resp.get("models", []) if m.get("model")]
except Exception as e:
    print(f"Error fetching models from Ollama: {e}")
    _OLLAMA_ALL_MODELS = []

# Vision-capable (multimodal) models we care about for VLM
OLLAMA_VLM_MODELS = [
    m for m in _OLLAMA_ALL_MODELS if any(x in m for x in ["gemma3", "llava", "vision"])
]

# Fallback static VLM list (HF)
HUGGINGFACE_VLM_MODELS = [
    "microsoft/git-base-coco",
    "microsoft/git-large-coco",
    "nlpconnect/vit-gpt2-image-captioning",
    "Salesforce/blip-image-captioning-base",
]

# Merge lists
VLM_MODELS = list(dict.fromkeys(OLLAMA_VLM_MODELS + HUGGINGFACE_VLM_MODELS))  # preserve order & unique

# LLM MODELS -> prefer Ollama list; fallback to static
if _OLLAMA_ALL_MODELS:
    LLM_MODELS = _OLLAMA_ALL_MODELS
else:
    LLM_MODELS = [
        "gemma3:latest",
        "gemma3:4b",
        "gemma3:2b",
        "gemma3:9b",
        "llama3.2:3b",
        "llama3.2:1b",
        "llama3.1:8b",
        "gemma2:9b",
        "mistral:7b",
    ]

# Determine default LLM and VLM models
DEFAULT_LLM_MODEL = "gemma3:latest" if "gemma3:latest" in LLM_MODELS else LLM_MODELS[0]
DEFAULT_VLM_MODEL = "gemma3:latest" if "gemma3:latest" in VLM_MODELS else VLM_MODELS[0]
# ------------------------

RETRIEVAL_STRATEGIES = [
    "hybrid",
    "text_only", 
    "pages_only"
]

DATA_PATH = "data"

# Global RAG system instance
rag_system: MultimodalRAGSystem = None

def initialize_rag_system(text_embedding_model: str, 
                         multimodal_embedding_model: str,
                         vlm_model: str,
                         llm_model: str,
                         jina_api_key: str = None):
    """Initialize the RAG system with given parameters."""
    global rag_system
    
    config = {
        "text_embedding_model": text_embedding_model,
        "multimodal_embedding_model": multimodal_embedding_model,
        "vlm_model": vlm_model,
        "llm_model": llm_model,
        "jina_api_key": jina_api_key if jina_api_key else None
    }
    
    rag_system = create_multimodal_rag_system(config)
    return rag_system

def process_query(
    question: str,
    text_embedding_model: str,
    multimodal_embedding_model: str,
    vlm_model: str,
    llm_model: str,
    retrieval_strategy: str,
    k_text: int,
    k_pages: int,
    include_images: bool,
    include_tables: bool,
    retrieved_content_state: list
) -> tuple:
    """Process a user query and return results."""
    
    if not rag_system:
        return ("Error: System not initialized. Please populate the database first.", 
                None, "❌ System not initialized", None, None)
    
    start_time = time.time()
    status_msg = "🔍 Processing multimodal query..."
    
    try:
        # Query the RAG system
        result = rag_system.query(
            question=question,
            retrieval_strategy=retrieval_strategy,
            k_text=k_text,
            k_pages=k_pages,
            include_images=include_images,
            include_tables=include_tables
        )
        
        response = result["response"]
        retrieved_content = result["retrieved_content"]
        sources = result["sources"]
        retrieval_stats = result["retrieval_stats"]
        
        # Create DataFrame for display
        df_data = []
        for i, content in enumerate(retrieved_content):
            content_type = content["type"]
            source = content["source"]
            score = f"{content['score']:.4f}"
            
            if content_type == "text":
                df_data.append([content_type, source, content["content"][:200] + "...", score])
            elif content_type in ["image", "graph"]:
                description = content.get("description", "No description")
                df_data.append([content_type, source, description[:200] + "...", score])
            elif content_type == "table":
                description = content.get("description", "No description")
                df_data.append([content_type, source, description[:200] + "...", score])
            elif content_type == "page_image":
                page = content.get("page", "Unknown")
                df_data.append([content_type, source, f"Page {page} image", score])
        
        # Create retrieval statistics display
        stats_text = f"""
        **Retrieval Statistics:**
        - Strategy: {retrieval_stats['strategy']}
        - Total Retrieved: {retrieval_stats['total_retrieved']}
        - By Type: {retrieval_stats['by_type']}
        """
        
        elapsed_time = time.time() - start_time
        status_msg = f"✅ Query processed in {elapsed_time:.2f}s | Sources: {', '.join(sources)}"
        
        return (response, 
                gr.Dataframe(
                    headers=['Type', 'Source', 'Content', 'Score'],
                    value=df_data
                ),
                status_msg,
                stats_text,
                retrieved_content)  # Pass this for potential export
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return (f"Error processing query: {str(e)}", 
                None, 
                f"❌ Error: {str(e)}", 
                None, 
                None)

def handle_file_upload(files, reset_db):
    """Handle file upload to data directory."""
    if not files:
        return "No files uploaded."
    
    # Create data directory if it doesn't exist
    os.makedirs(DATA_PATH, exist_ok=True)
    
    # Copy uploaded files to the data directory
    file_count = 0
    for file in files:
        try:
            filename = Path(file.name).name
            destination = os.path.join(DATA_PATH, filename)
            copy2(file.name, destination)
            file_count += 1
        except Exception as e:
            return f"Error copying file {file.name}: {str(e)}"
    
    return f"✅ Successfully uploaded {file_count} files to the data directory."

def populate_database(
    reset_db: bool,
    text_embedding_model: str,
    multimodal_embedding_model: str,
    vlm_model: str,
    llm_model: str,
    jina_api_key: str = None
):
    """Populate the database with documents from data directory."""
    global rag_system
    
    try:
        # Initialize or reinitialize the system
        rag_system = initialize_rag_system(
            text_embedding_model=text_embedding_model,
            multimodal_embedding_model=multimodal_embedding_model,
            vlm_model=vlm_model,
            llm_model=llm_model,
            jina_api_key=jina_api_key
        )
        
        if reset_db:
            rag_system.clear_database()
        
        # Get list of files to process
        if not os.path.exists(DATA_PATH):
            return "❌ Data directory not found. Please upload files first."
        
        file_paths = []
        for ext in [".pdf", ".docx", ".doc"]:
            file_paths.extend(Path(DATA_PATH).glob(f"*{ext}"))
        
        if not file_paths:
            return "❌ No supported files found in data directory."
        
        file_paths = [str(p) for p in file_paths]
        
        # Process documents
        processing_stats = rag_system.process_documents(file_paths)
        
        # Get database stats
        db_stats = rag_system.get_database_stats()
        
        result_msg = f"""
✅ Database populated successfully!

**Processing Statistics:**
- Files processed: {processing_stats['files_processed']}
- Total elements extracted: {processing_stats['total_elements']}
- Total documents created: {processing_stats['total_documents']}
- Elements by type: {processing_stats['elements_by_type']}

**Database Statistics:**
- Text content documents: {db_stats['text_content']['count']}
- Page image documents: {db_stats['page_images']['count']}

**Errors:** {len(processing_stats['errors'])}
{chr(10).join(processing_stats['errors']) if processing_stats['errors'] else 'None'}
        """
        
        return result_msg
        
    except Exception as e:
        logger.error(f"Error populating database: {e}")
        return f"❌ Error: {str(e)}"

def export_retrieved_content(retrieved_content, output_format):
    """Export retrieved content for download."""
    if not retrieved_content:
        return None, "No content to export."
    
    try:
        export_dir = "exported_content"
        exported_files = rag_system.export_retrieved_content(retrieved_content, export_dir)
        
        export_summary = f"""
**Exported Content:**
- Images: {len(exported_files['images'])} files
- Tables: {len(exported_files['tables'])} files  
- Texts: {len(exported_files['texts'])} files

Files saved to: {export_dir}/
        """
        
        return export_summary, "✅ Content exported successfully!"
        
    except Exception as e:
        return None, f"❌ Export failed: {str(e)}"

# Create the Gradio interface
with gr.Blocks(title="Multimodal RAG System", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🔬 Multimodal RAG System
        
        Advanced document Q&A system that handles **text, images, tables, graphs, and page-level content** from PDF and DOCX files.
        
        **Features:**
        - Multimodal content extraction and processing
        - Separate embeddings for text and visual content  
        - Hybrid retrieval from multiple collections
        - Vision Language Model descriptions for images
        - HTML table structure preservation
        """
    )
    
    # Store retrieved content for export
    retrieved_content_state = gr.State([])
    
    with gr.Tab("🔍 Query Documents"):
        with gr.Row():
            with gr.Column(scale=2):
                query_input = gr.Textbox(
                    label="Enter your question",
                    placeholder="What do the financial charts show? Describe the tables in the report.",
                    lines=4
                )
            with gr.Column(scale=1):
                status_display = gr.Textbox(label="Status", interactive=False, lines=3)
        
        query_button = gr.Button("Submit Query", variant="primary", scale=1)
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 🤖 AI Response")
                output = gr.Textbox(label="Response", lines=6)
        
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### ⚙️ Model Settings")
                    with gr.Row():
                        with gr.Column():
                            llm_model = gr.Dropdown(
                                choices=LLM_MODELS,
                                value=DEFAULT_LLM_MODEL,
                                label="💬 LLM Model",
                                info="Language model for response generation"
                            )
                            
                            text_embedding_model = gr.Dropdown(
                                choices=EMBEDDING_MODELS,
                                value="jinaai/jina-embeddings-v3",
                                label="📝 Text Embedding Model",
                                info="Model for embedding text content"
                            )
                        
                        with gr.Column():
                            multimodal_embedding_model = gr.Dropdown(
                                choices=MULTIMODAL_EMBEDDING_MODELS,
                                value="jinaai/jina-embeddings-v4",
                                label="🖼️ Multimodal Embedding Model",
                                info="Model for embedding page images"
                            )
                            
                            vlm_model = gr.Dropdown(
                                choices=VLM_MODELS,
                                value=DEFAULT_VLM_MODEL,
                                label="👁️ Vision Language Model",
                                info="Model for describing images and graphs"
                            )
            
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 🎯 Retrieval Settings")
                    retrieval_strategy_dropdown = gr.Dropdown(
                        choices=RETRIEVAL_STRATEGIES,
                        value=RETRIEVAL_STRATEGIES[0],
                        label="Retrieval Strategy",
                        info="How to search the database"
                    )
                    
                    k_text_slider = gr.Slider(
                        minimum=1, maximum=20, value=8, step=1,
                        label="Text Results (k_text)",
                        info="Number of text content results"
                    )
                    
                    k_pages_slider = gr.Slider(
                        minimum=1, maximum=10, value=2, step=1,
                        label="Page Results (k_pages)", 
                        info="Number of page image results"
                    )
                    
                    include_images_checkbox = gr.Checkbox(
                        label="Include Images/Graphs",
                        value=True,
                        info="Include image descriptions in response"
                    )
                    
                    include_tables_checkbox = gr.Checkbox(
                        label="Include Tables",
                        value=True,
                        info="Include table descriptions in response"
                    )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Retrieval Statistics")
                stats_output = gr.Markdown()
            
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Export Results")
                export_button = gr.Button("Export Retrieved Content", variant="secondary")
                export_status = gr.Textbox(label="Export Status", interactive=False, lines=2)
        
        with gr.Row():
            gr.Markdown("### 📋 Retrieved Content")
            chunks_output = gr.Dataframe(
                headers=['Type', 'Source', 'Content', 'Score'],
                label="Retrieved Documents",
                wrap=True,
                column_widths=[10, 20, 50, 10]
            )
        
        # API Key input (optional)
        with gr.Row():
            jina_api_key_input = gr.Textbox(
                label="Jina API Key (Optional)",
                placeholder="Enter Jina API key for enhanced multimodal embeddings",
                type="password"
            )
        
        query_button.click(
            fn=process_query,
            inputs=[
                query_input,
                text_embedding_model,
                multimodal_embedding_model,
                vlm_model,
                llm_model,
                retrieval_strategy_dropdown,
                k_text_slider,
                k_pages_slider,
                include_images_checkbox,
                include_tables_checkbox,
                retrieved_content_state
            ],
            outputs=[output, chunks_output, status_display, stats_output, retrieved_content_state]
        )
        
        export_button.click(
            fn=export_retrieved_content,
            inputs=[retrieved_content_state, gr.State("zip")],
            outputs=[export_status, gr.State()]
        )
    
    with gr.Tab("📁 Document Management"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📤 Upload Documents")
                file_upload = gr.File(
                    file_types=[".pdf", ".docx", ".doc"],
                    file_count="multiple",
                    label="Upload Files (PDF, DOCX)"
                )
                upload_button = gr.Button("Upload Files", variant="primary")
                upload_status = gr.Textbox(label="Upload Status", interactive=False)
            
            with gr.Column():
                gr.Markdown("### 🗄️ Database Control")
                with gr.Group():
                    reset_checkbox = gr.Checkbox(
                        label="Reset Database",
                        value=False,
                        info="Clear existing database before processing"
                    )
                    
                    # Database model settings
                    db_text_embedding_dropdown = gr.Dropdown(
                        choices=EMBEDDING_MODELS,
                        value="jinaai/jina-embeddings-v3",
                        label="Text Embedding Model"
                    )
                    
                    db_multimodal_embedding_dropdown = gr.Dropdown(
                        choices=MULTIMODAL_EMBEDDING_MODELS,
                        value="jinaai/jina-embeddings-v4",
                        label="Multimodal Embedding Model"
                    )
                    
                    db_vlm_dropdown = gr.Dropdown(
                        choices=VLM_MODELS,
                        value=DEFAULT_VLM_MODEL,
                        label="Vision Language Model"
                    )
                    
                    db_llm_dropdown = gr.Dropdown(
                        choices=LLM_MODELS,
                        value=DEFAULT_LLM_MODEL,
                        label="LLM Model"
                    )
                    
                    db_jina_api_key_input = gr.Textbox(
                        label="Jina API Key (Optional)",
                        placeholder="For enhanced multimodal embeddings",
                        type="password"
                    )
                    
                    populate_button = gr.Button("Populate Database", variant="primary")
                    status_output = gr.Textbox(label="Processing Status", interactive=False, lines=8)
        
        upload_button.click(
            fn=handle_file_upload,
            inputs=[file_upload, reset_checkbox],
            outputs=upload_status
        )
        
        populate_button.click(
            fn=populate_database,
            inputs=[
                reset_checkbox,
                db_text_embedding_dropdown,
                db_multimodal_embedding_dropdown,
                db_vlm_dropdown,
                db_llm_dropdown,
                db_jina_api_key_input
            ],
            outputs=status_output
        )
    
    with gr.Tab("ℹ️ About"):
        gr.Markdown(
            """
            ## 🔬 About Multimodal RAG System
            
            This system combines advanced document processing with multimodal AI to provide comprehensive answers based on your documents.
            
            ### ✨ Key Features
            
            **📄 Document Processing:**
            - PDF and DOCX support
            - Automatic content type detection
            - Page-level image extraction
            
            **🧠 Multimodal Understanding:**
            - **Text**: Traditional text chunking and embedding
            - **Images**: Vision Language Model descriptions + original image retrieval
            - **Graphs**: Treated as images with specialized descriptions
            - **Tables**: HTML structure preservation + LLM-generated descriptions
            - **Pages**: Full page images with multimodal embeddings
            
            **🔍 Advanced Retrieval:**
            - Dual collection system (text content + page images)
            - Hybrid search across content types
            - Configurable retrieval strategies
            - Multimodal embedding support (Jina v4, CLIP)
            
            **🎯 Smart Response Generation:**
            - Context-aware responses referencing specific content types
            - Image and table descriptions in answers
            - Source attribution by content type
            - Export functionality for retrieved content
            
            ### 🛠️ How It Works
            
            1. **Upload** your PDF/DOCX documents
            2. **Process** documents to extract text, images, tables, and graphs
            3. **Embed** different content types with appropriate models
            4. **Store** in specialized ChromaDB collections  
            5. **Query** with natural language questions
            6. **Retrieve** relevant multimodal content
            7. **Generate** comprehensive responses using LLM
            
            ### 🔧 Technical Architecture
            
            - **Document Processor**: PyMuPDF, python-docx, PIL
            - **Vision Models**: Git-COCO, BLIP, ViT-GPT2
            - **Embeddings**: Jina v3/v4, E5, Turkish models, CLIP
            - **Vector DB**: ChromaDB with dual collections
            - **LLM**: Ollama (Llama, Gemma, Mistral)
            - **Interface**: Gradio web UI
            
            For best results, use specific questions about visual content like charts, graphs, and tables!
            """
        )

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860) 