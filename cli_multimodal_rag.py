#!/usr/bin/env python3
"""
Multimodal RAG CLI - Using the same logic as multimodal_app.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# std‑lib
import os
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# third‑party
import requests
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.markdown import Markdown

# Import the same components as multimodal_app.py
from multimodal_rag_system import create_multimodal_rag_system
from config import get_cli_config

console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MultimodalRAG_CLI")

# ═════════════════════════════════════════════════════════════════════════════
# Same functions as multimodal_app.py but adapted for CLI
# ═════════════════════════════════════════════════════════════════════════════

def initialize_system(config: dict = None):
    """Initialize the multimodal RAG system - same logic as multimodal_app.py"""
    if config is None:
        config = get_cli_config()
    
    console.print("🔧 [blue]Initializing multimodal RAG system...[/blue]")
    
    try:
        rag_system = create_multimodal_rag_system(config)
        console.print("✅ [green]System initialized successfully![/green]")
        return rag_system
    except Exception as e:
        console.print(f"❌ [red]System initialization failed: {e}[/red]")
        raise

def process_documents(rag_system, file_paths: List[str], reset_db: bool = False):
    """Process documents - same logic as multimodal_app.py handle_file_upload"""
    try:
        if reset_db:
            console.print("🗑️ [yellow]Clearing database...[/yellow]")
            rag_system.clear_database()
        
        console.print(f"📁 [blue]Processing {len(file_paths)} files...[/blue]")
        
        results = rag_system.process_documents(file_paths)
        
        console.print("✅ [green]Documents processed successfully![/green]")
        console.print(f"📊 Results: {results}")
        
        return results
        
    except Exception as e:
        console.print(f"❌ [red]Document processing failed: {e}[/red]")
        raise

def query_system(rag_system, question: str, retrieval_strategy: str = "hybrid", 
                k_text: int = 8, k_pages: int = 2, include_images: bool = True, 
                include_tables: bool = True):
    """Query the system - same logic as multimodal_app.py process_query"""
    
    console.print(f"🔍 [blue]Processing query with strategy: {retrieval_strategy}[/blue]")
    
    try:
        result = rag_system.query(
            question=question,
            retrieval_strategy=retrieval_strategy,
            k_text=k_text,
            k_pages=k_pages,
            include_images=include_images,
            include_tables=include_tables
        )
        
        return result
        
    except Exception as e:
        console.print(f"❌ [red]Query processing failed: {e}[/red]")
        raise

def display_results(result: dict):
    """Display query results in a nice format"""
    
    # Show response
    console.print(Panel(
        Markdown(result.get("response", "No response")), 
        title="🤖 Cevap (Answer)", 
        border_style="green"
    ))
    
    # Show retrieval stats
    stats = result.get("retrieval_stats", {})
    if stats:
        console.print(f"📊 [cyan]Retrieved:[/cyan] {stats.get('total_retrieved', 0)} documents")
        console.print(f"🔍 [cyan]Strategy:[/cyan] {stats.get('strategy', 'unknown')}")
        
        by_type = stats.get('by_type', {})
        if by_type:
            console.print("📋 [cyan]By type:[/cyan]")
            for doc_type, count in by_type.items():
                console.print(f"   • {doc_type}: {count}")
    
    # Show processing time
    processing_time = result.get("processing_time", 0)
    console.print(f"⏱️ [cyan]Processing time:[/cyan] {processing_time:.2f}s")
    
    # Show sources
    sources = result.get("sources", [])
    if sources:
        console.print(f"\n📚 [yellow]Sources ({len(sources)}):[/yellow]")
        for i, source in enumerate(sources[:5], 1):  # Show first 5 sources
            source_file = source #source.get("source_file", "Unknown")
            page = source #source.get("page", "?")
            element_type = source #source.get("element_type", "unknown")
            console.print(f"   {i}. {source_file} (Page {page}, {element_type})")

def get_database_stats(rag_system):
    """Get database statistics - same as multimodal_app.py"""
    try:
        stats = rag_system.get_database_stats()
        console.print("\n📊 [cyan]Database Statistics:[/cyan]")
        
        text_stats = stats.get("text_content", {})
        page_stats = stats.get("page_images", {})
        
        console.print(f"   📝 Text content: {text_stats.get('count', 0)} documents")
        console.print(f"   🖼️ Page images: {page_stats.get('count', 0)} documents")
        
        return stats
    except Exception as e:
        console.print(f"❌ [red]Error getting database stats: {e}[/red]")
        return {}

# ═════════════════════════════════════════════════════════════════════════════
# CLI Interface
# ═════════════════════════════════════════════════════════════════════════════

def test_jina_embeddings(jina_api_url: str):
    """Test Jina embeddings API similar to user's test code"""
    console.print("\n🧪 [cyan]Testing Jina Embeddings API...[/cyan]")
    
    # Test text embedding
    test_text = "Merhaba dünya, bu bir test metnidir."
    payload = {
        "model": "jinaai/jina-embeddings-v4",
        "texts": [test_text],
        "task": "retrieval",
        "prompt_name": "query"
    }
    
    try:
        response = requests.post(
            f"{jina_api_url}/embed/text",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            console.print("✓ [green]Text embedding test successful![/green]")
            console.print(f"   Model: {result['model']}")
            console.print(f"   Shape: {result['shape']}")
            console.print(f"   Task: {result['task']}")
            return True
        else:
            console.print(f"✗ [red]Text embedding test failed: {response.status_code}[/red]")
            return False
            
    except Exception as e:
        console.print(f"✗ [red]Text embedding test error: {e}[/red]")
        return False

def test_connections(jina_api_url: str, ollama_base_url: str):
    """Test API connections"""
    console.print("\n🔍 Testing connections...")
    
    # Test Jina API (if using)
    jina_ok = True
    try:
        if jina_api_url and "10.144.100.204" in jina_api_url:
            response = requests.get(f"{jina_api_url}/health", timeout=5)
            if response.status_code == 200:
                console.print("✓ [green]Jina API is running[/green]")
            else:
                console.print(f"✗ [red]Jina API error: {response.status_code}[/red]")
                jina_ok = False
    except:
        console.print("ℹ️ [yellow]Jina API not accessible (will use fallback)[/yellow]")
        jina_ok = False
    
    # Test Ollama
    ollama_ok = True
    try:
        response = requests.get(f"{ollama_base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            console.print("✓ [green]Ollama API is running[/green]")
        else:
            console.print(f"✗ [red]Ollama API error: {response.status_code}[/red]")
            ollama_ok = False
    except:
        console.print("✗ [red]Cannot connect to Ollama API[/red]")
        ollama_ok = False
    
    return jina_ok, ollama_ok

def interactive_file_selection():
    """Interactive file selection for all supported formats"""
    data_folder = Path("data")
    if not data_folder.exists():
        console.print("[red]Data folder not found. Creating it...[/red]")
        data_folder.mkdir(exist_ok=True)
        console.print("[yellow]Please add files to the 'data' folder and restart.[/yellow]")
        return []
    
    # Supported file extensions
    supported_extensions = [
        '.pdf', '.doc', '.docx',           # Documents
        '.txt', '.rtf',                    # Text files  
        '.csv', '.xls', '.xlsx',           # Spreadsheets
        '.json',                           # Data files
        '.md', '.markdown',                # Markdown
        '.html', '.htm',                   # HTML
        '.jpg', '.jpeg', '.png', '.gif',   # Images
        '.bmp', '.tiff', '.webp'
    ]
    
    # Find all supported files
    supported_files = []
    for ext in supported_extensions:
        supported_files.extend(list(data_folder.glob(f"*{ext}")))
        supported_files.extend(list(data_folder.glob(f"*{ext.upper()}")))
    
    if not supported_files:
        console.print(f"[red]No supported files found in data folder.[/red]")
        console.print(f"[cyan]Supported formats:[/cyan]")
        console.print(f"   📄 Documents: PDF, DOC, DOCX")
        console.print(f"   📝 Text: TXT, RTF") 
        console.print(f"   📊 Spreadsheets: CSV, XLS, XLSX")
        console.print(f"   📋 Data: JSON")
        console.print(f"   📖 Markup: MD, HTML")
        console.print(f"   🖼️ Images: JPG, PNG, GIF, BMP, TIFF, WEBP")
        return []
    
    # Group files by type
    file_groups = {
        'Documents': [f for f in supported_files if f.suffix.lower() in ['.pdf', '.doc', '.docx']],
        'Text Files': [f for f in supported_files if f.suffix.lower() in ['.txt', '.rtf']],
        'Spreadsheets': [f for f in supported_files if f.suffix.lower() in ['.csv', '.xls', '.xlsx']],
        'Data Files': [f for f in supported_files if f.suffix.lower() in ['.json']],
        'Markup': [f for f in supported_files if f.suffix.lower() in ['.md', '.markdown', '.html', '.htm']],
        'Images': [f for f in supported_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']]
    }
    
    console.print(f"\n📁 [cyan]Found {len(supported_files)} supported files:[/cyan]")
    for group_name, files in file_groups.items():
        if files:
            console.print(f"\n[yellow]{group_name}:[/yellow]")
            for i, file in enumerate(files, 1):
                console.print(f"   {i}. {file.name}")
    
    choice = Prompt.ask(f"\n[yellow]Process all {len(supported_files)} files? (y/n)[/yellow]", default="y")
    if choice.lower() in ['y', 'yes']:
        return [str(f) for f in supported_files]
    else:
        return []

def main():
    console.print(Panel("🔬 Multimodal RAG CLI - Same Logic as Web App", expand=False, border_style="blue"))
    
    # Get CLI-specific configuration
    config = get_cli_config()
    
    console.print(f"[yellow]Configuration:[/yellow]")
    console.print(f"   🗄️ Database: {config['chroma_path']}")
    console.print(f"   📁 Extracted content: {config['extracted_content_path']}")
    console.print(f"   📝 Text embedding: {config['text_embedding_model']}")
    console.print(f"   🖼️ Multimodal embedding: {config['multimodal_embedding_model']}")
    console.print(f"   👁️ VLM model: {config['vlm_model']}")
    console.print(f"   🤖 LLM model: {config['llm_model']}")
    console.print(f"   🔑 Jina API key: {'✓ Set' if config['jina_api_key'] else '✗ Not set'}")
    console.print(f"   🌐 API base URL: {config['jina_api_base_url'] or 'Not set'}")
    console.print(f"   🏠 Force local: {'Yes' if config['force_local_embeddings'] else 'No'}")
    
    # Test connections
    jina_api_url = config['jina_api_base_url']
    ollama_base_url = config['ollama_base_url']
    
    console.print(f"\n[yellow]API Endpoints:[/yellow]")
    console.print(f"   🔗 Jina API: {jina_api_url}")
    console.print(f"   🔗 Ollama API: {ollama_base_url}")
    
    jina_ok, ollama_ok = test_connections(jina_api_url, ollama_base_url)
    
    # Test Jina embeddings API if connection is ok
    if jina_ok:
        jina_embed_ok = test_jina_embeddings(jina_api_url)
        if not jina_embed_ok:
            console.print("[yellow]Jina embeddings test failed, but system will use fallback[/yellow]")
    
    if not ollama_ok:
        console.print("[red]Ollama connection failed. Please check if Ollama is running.[/red]")
        return
    
    # Initialize system - same as multimodal_app.py
    try:
        rag_system = initialize_system(config)
    except Exception as e:
        console.print(f"[red]Failed to initialize system: {e}[/red]")
        return
    
    # Get database stats
    get_database_stats(rag_system)
    
    # Check if we need to process documents
    stats = rag_system.get_database_stats()
    total_docs = stats.get("text_content", {}).get("count", 0) + stats.get("page_images", {}).get("count", 0)
    
    if total_docs == 0:
        console.print("\n📂 [yellow]No documents in database. Let's add some![/yellow]")
        file_paths = interactive_file_selection()
        
        if file_paths:
            try:
                process_documents(rag_system, file_paths, reset_db=False)
                get_database_stats(rag_system)
            except Exception as e:
                console.print(f"[red]Failed to process documents: {e}[/red]")
                return
        else:
            console.print("[yellow]No files to process. You can still test with empty database.[/yellow]")
    
    console.print("\n🎯 [bold green]System ready! Type 'quit' to exit.[/bold green]")
    console.print("💡 [dim]Available commands: quit, exit, q, stats, help[/dim]\n")
    
    # Main interaction loop - same query logic as multimodal_app.py
    while True:
        try:
            query = Prompt.ask("\n[bold yellow]Soru (Question)[/bold yellow]").strip()
            
            if query.lower() in ["quit", "exit", "q"]:
                break
            elif query.lower() == "stats":
                get_database_stats(rag_system)
                continue
            elif query.lower() == "help":
                console.print("""
[cyan]🔬 Multimodal RAG CLI - Comprehensive File Format Support[/cyan]

[yellow]Available Commands:[/yellow]
• Type your question to query the documents
• 'stats' - Show database statistics  
• 'quit', 'exit', 'q' - Exit the application
• 'help' - Show this help message

[yellow]Supported File Formats:[/yellow]
📄 [bold]Documents:[/bold]
   • PDF (.pdf) - Full multimodal processing with text, images, tables, and pages
   • Word (.doc, .docx) - Complete document structure extraction
   
📝 [bold]Text Files:[/bold] 
   • Plain Text (.txt) - Recursive text splitting and embedding
   • Rich Text (.rtf) - RTF to text conversion with processing
   
📊 [bold]Spreadsheets:[/bold]
   • CSV (.csv) - Convert to HTML tables with descriptions
   • Excel (.xls, .xlsx) - Multi-sheet processing as table elements
   
📋 [bold]Data Files:[/bold]
   • JSON (.json) - Smart parsing with table detection for arrays
   
📖 [bold]Markup:[/bold]
   • Markdown (.md) - Hierarchy-aware splitting with image extraction
   • HTML (.html, .htm) - Structure-preserving text and image extraction
   
🖼️ [bold]Images:[/bold]
   • Photos/Graphics (.jpg, .jpeg, .png, .gif, .bmp, .tiff, .webp)
   • Automatic graph/chart detection and description generation

[yellow]Processing Features:[/yellow]
• 🧠 VLM-powered image descriptions with context
• 📊 LLM-generated table summaries  
• 🔍 Hierarchy-aware text splitting (MD/HTML)
• 🖼️ Dual embedding strategy (text + multimodal)
• 📑 Page-level image embeddings for visual search

[yellow]Query Settings (CLI):[/yellow]
• Retrieval strategy: hybrid (text + images)
• Text chunks: 8
• Page images: 2  
• Include images: Yes
• Include tables: Yes
                """)
                continue
            elif not query:
                continue

            console.print(f"\n🔍 [blue]Processing query:[/blue] {query}")
            
            # Query with same parameters as multimodal_app.py defaults
            result = query_system(
                rag_system=rag_system,
                question=query,
                retrieval_strategy="hybrid",  # Same as multimodal_app.py default
                k_text=8,                     # Same as multimodal_app.py default
                k_pages=2,                    # Same as multimodal_app.py default
                include_images=True,          # Same as multimodal_app.py default
                include_tables=True           # Same as multimodal_app.py default
            )
            
            # Display results
            display_results(result)

        except (KeyboardInterrupt, EOFError):
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            import traceback
            traceback.print_exc()
    
    console.print("\n👋 [yellow]Goodbye![/yellow]")

if __name__ == "__main__":
    main() 