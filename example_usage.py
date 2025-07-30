#!/usr/bin/env python3
"""
Example usage of the Multimodal RAG System

This script demonstrates how to use the multimodal RAG system programmatically
without the web interface.
"""

import os
import logging
from pathlib import Path

# Import our multimodal RAG system
try:
    from multimodal_rag_system import create_multimodal_rag_system
except ImportError:
    from ultimodal_rag_system import create_multimodal_rag_system

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main example function demonstrating the multimodal RAG system."""
    
    print("🔬 Multimodal RAG System - Example Usage")
    print("=" * 50)
    
    # Configuration for the RAG system
    config = {
        "chroma_path": "example_chroma",
        "extracted_content_path": "example_extracted_content",
        "text_embedding_model": "jinaai/jina-embeddings-v3",
        "multimodal_embedding_model": "jinaai/jina-embeddings-v4",
        "vlm_model": "microsoft/git-base-coco",
        "llm_model": "llama3.2:3b",
        "jina_api_key": os.getenv("JINA_API_KEY")  # Optional
    }
    
    print(f"📋 Configuration:")
    for key, value in config.items():
        if key == "jina_api_key":
            print(f"  {key}: {'Set' if value else 'Not set'}")
        else:
            print(f"  {key}: {value}")
    print()
    
    # Initialize the RAG system
    print("🚀 Initializing Multimodal RAG System...")
    rag_system = create_multimodal_rag_system(config)
    print("✅ System initialized successfully!")
    print()
    
    # Example document processing
    print("📁 Processing Documents...")
    
    # You would replace this with your actual document paths
    document_paths = [
        # "path/to/your/document1.pdf",
        # "path/to/your/document2.docx",
    ]
    
    # Check if we have documents to process
    if not document_paths:
        print("⚠️  No documents specified. Please add document paths to the script.")
        print("   You can add PDF or DOCX files to process.")
        print()
        
        # Show example with fake stats
        example_stats = {
            "files_processed": 0,
            "total_elements": 0,
            "total_documents": 0,
            "elements_by_type": {},
            "errors": []
        }
        print(f"📊 Processing would show stats like: {example_stats}")
        print()
    else:
        # Process documents
        processing_stats = rag_system.process_documents(document_paths)
        
        print(f"📊 Processing Statistics:")
        print(f"  Files processed: {processing_stats['files_processed']}")
        print(f"  Total elements: {processing_stats['total_elements']}")
        print(f"  Total documents: {processing_stats['total_documents']}")
        print(f"  Elements by type: {processing_stats['elements_by_type']}")
        
        if processing_stats['errors']:
            print(f"  Errors: {len(processing_stats['errors'])}")
            for error in processing_stats['errors']:
                print(f"    - {error}")
        print()
    
    # Get database statistics
    print("📈 Database Statistics:")
    db_stats = rag_system.get_database_stats()
    print(f"  Text content documents: {db_stats['text_content']['count']}")
    print(f"  Page image documents: {db_stats['page_images']['count']}")
    print()
    
    # Example queries
    example_queries = [
        "What are the main topics discussed in the documents?",
        "Are there any charts or graphs? What do they show?",
        "Summarize the key data from any tables in the documents.",
        "What visual information is available on the first few pages?",
    ]
    
    print("❓ Example Queries:")
    for i, query in enumerate(example_queries, 1):
        print(f"  {i}. {query}")
    print()
    
    # If no documents, show what a query would look like
    if not document_paths or db_stats['text_content']['count'] == 0:
        print("💡 Example Query (would need documents in database):")
        example_query = "What are the main financial metrics shown in the quarterly report?"
        print(f"   Query: '{example_query}'")
        print("   Would return:")
        print("   - Relevant text chunks")
        print("   - Image descriptions")
        print("   - Table summaries")
        print("   - Page-level visual content")
        print("   - AI-generated comprehensive answer")
        print()
        return
    
    # Perform example query
    example_query = "What are the main topics discussed in the documents?"
    print(f"🔍 Performing example query: '{example_query}'")
    
    try:
        result = rag_system.query(
            question=example_query,
            retrieval_strategy="hybrid",
            k_text=5,
            k_pages=2,
            include_images=True,
            include_tables=True
        )
        
        print("📋 Query Results:")
        print(f"  Response: {result['response'][:200]}...")
        print(f"  Sources: {', '.join(result['sources'])}")
        print(f"  Processing time: {result['processing_time']:.2f}s")
        print(f"  Retrieved content items: {len(result['retrieved_content'])}")
        
        # Show retrieval statistics
        stats = result['retrieval_stats']
        print(f"  Retrieval stats:")
        print(f"    Strategy: {stats['strategy']}")
        print(f"    Total retrieved: {stats['total_retrieved']}")
        print(f"    By type: {stats['by_type']}")
        
        # Show retrieved content details
        print("\n📄 Retrieved Content Details:")
        for i, content in enumerate(result['retrieved_content'][:3]):  # Show first 3
            print(f"  {i+1}. Type: {content['type']}")
            print(f"     Source: {content['source']}")
            print(f"     Score: {content['score']:.4f}")
            if content['type'] == 'text':
                print(f"     Content: {content['content'][:100]}...")
            elif content['type'] in ['image', 'graph']:
                print(f"     Description: {content.get('description', 'N/A')[:100]}...")
            elif content['type'] == 'table':
                print(f"     Description: {content.get('description', 'N/A')[:100]}...")
            elif content['type'] == 'page_image':
                print(f"     Page: {content.get('page', 'Unknown')}")
            print()
        
        # Export example
        print("💾 Export Example:")
        export_dir = "example_export"
        exported_files = rag_system.export_retrieved_content(
            result['retrieved_content'], 
            export_dir
        )
        print(f"  Exported to: {export_dir}/")
        print(f"  Images: {len(exported_files['images'])} files")
        print(f"  Tables: {len(exported_files['tables'])} files")
        print(f"  Texts: {len(exported_files['texts'])} files")
        
    except Exception as e:
        print(f"❌ Error during query: {e}")
        print("   This might happen if:")
        print("   - Ollama is not running")
        print("   - Required models are not available")
        print("   - No documents have been processed")
    
    print("\n🎉 Example completed!")
    print("\nNext steps:")
    print("1. Add your PDF/DOCX files to the document_paths list")
    print("2. Run the script again to process real documents")
    print("3. Try different queries and retrieval strategies")
    print("4. Use the web interface (multimodal_app.py) for easier interaction")

def demonstrate_different_strategies():
    """Demonstrate different retrieval strategies."""
    print("\n🎯 Retrieval Strategy Examples:")
    
    strategies = {
        "hybrid": "Search both text content and page images (recommended)",
        "text_only": "Search only text, descriptions, and tables",
        "pages_only": "Search only page-level visual content"
    }
    
    for strategy, description in strategies.items():
        print(f"  {strategy}: {description}")
    
    print("\n📊 Content Type Filtering:")
    filters = {
        "include_images": "Include image and graph descriptions",
        "include_tables": "Include table descriptions and structure"
    }
    
    for filter_name, description in filters.items():
        print(f"  {filter_name}: {description}")

def show_model_options():
    """Show available model options."""
    print("\n🧠 Available Models:")
    
    models = {
        "Text Embeddings": [
            "jinaai/jina-embeddings-v3",
            "intfloat/multilingual-e5-large-instruct",
            "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
        ],
        "Multimodal Embeddings": [
            "jinaai/jina-embeddings-v4 (with API key)",
            "openai/clip-vit-base-patch32 (local)"
        ],
        "Vision Language Models": [
            "microsoft/git-base-coco",
            "Salesforce/blip-image-captioning-base",
            "nlpconnect/vit-gpt2-image-captioning"
        ],
        "Language Models (Ollama)": [
            "llama3.2:3b",
            "llama3.2:1b",
            "gemma2:9b",
            "mistral:7b"
        ]
    }
    
    for category, model_list in models.items():
        print(f"  {category}:")
        for model in model_list:
            print(f"    - {model}")
        print()

if __name__ == "__main__":
    main()
    demonstrate_different_strategies()
    show_model_options() 