#!/usr/bin/env python3
"""
Test script for the Multimodal RAG System

This script performs basic tests to ensure the system is working correctly.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        from multimodal_rag_system import create_multimodal_rag_system
        print("✅ multimodal_rag_system imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import multimodal_rag_system: {e}")
        return False
    
    try:
        from multimodal_document_processor import MultimodalDocumentProcessor
        print("✅ multimodal_document_processor imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import document processor: {e}")
        return False
    
    try:
        from multimodal_embeddings import MultimodalEmbeddingManager
        print("✅ multimodal_embeddings imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import embeddings: {e}")
        return False
    
    try:
        from multimodal_vector_db import MultimodalVectorDatabase
        print("✅ multimodal_vector_db imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import vector database: {e}")
        return False
    
    return True

def test_system_initialization():
    """Test that the system can be initialized."""
    print("\n🧪 Testing system initialization...")
    
    try:
        from multimodal_rag_system import create_multimodal_rag_system
        
        config = {
            "chroma_path": "test_chroma",
            "extracted_content_path": "test_extracted",
            "text_embedding_model": "jinaai/jina-embeddings-v3",
            "multimodal_embedding_model": "jinaai/jina-embeddings-v4",
            "vlm_model": "microsoft/git-base-coco",
            "llm_model": "llama3.2:3b",
            "jina_api_key": None
        }
        
        rag_system = create_multimodal_rag_system(config)
        print("✅ System initialized successfully")
        
        # Test database stats
        stats = rag_system.get_database_stats()
        print(f"✅ Database stats retrieved: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return False

def test_document_processor():
    """Test the document processor functionality."""
    print("\n🧪 Testing document processor...")
    
    try:
        from multimodal_document_processor import MultimodalDocumentProcessor
        
        processor = MultimodalDocumentProcessor()
        print("✅ Document processor created successfully")
        
        # Test that we can call methods without errors (no actual files)
        try:
            # This will fail with file not found, but should not have import/syntax errors
            processor.process_document("nonexistent.pdf")
        except (FileNotFoundError, ValueError) as e:
            print("✅ Document processor methods callable (expected file error)")
        except Exception as e:
            print(f"❌ Unexpected error in document processor: {e}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Failed to test document processor: {e}")
        return False

def test_embeddings():
    """Test the embeddings functionality."""
    print("\n🧪 Testing embeddings...")
    
    try:
        from multimodal_embeddings import MultimodalEmbeddingManager
        
        # Test with fallback (no API key)
        manager = MultimodalEmbeddingManager(
            text_embedding_model="jinaai/jina-embeddings-v3",
            multimodal_embedding_model="jinaai/jina-embeddings-v4",
            jina_api_key=None
        )
        print("✅ Embedding manager created successfully")
        
        # Test getting embedding functions
        text_embeddings = manager.get_text_embedding_function()
        multimodal_embeddings = manager.get_multimodal_embedding_function()
        print("✅ Embedding functions retrieved successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to test embeddings: {e}")
        return False

def test_vector_db():
    """Test the vector database functionality."""
    print("\n🧪 Testing vector database...")
    
    try:
        from multimodal_vector_db import MultimodalVectorDatabase
        
        vector_db = MultimodalVectorDatabase(
            chroma_path="test_chroma_db",
            text_embedding_model="jinaai/jina-embeddings-v3",
            multimodal_embedding_model="jinaai/jina-embeddings-v4",
            jina_api_key=None
        )
        print("✅ Vector database created successfully")
        
        # Test getting stats
        stats = vector_db.get_collection_stats()
        print(f"✅ Vector database stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to test vector database: {e}")
        return False

def check_dependencies():
    """Check if key dependencies are available."""
    print("\n🧪 Checking dependencies...")
    
    dependencies = [
        ("torch", "PyTorch for neural networks"),
        ("transformers", "Hugging Face Transformers"),
        ("langchain", "LangChain framework"),
        ("chromadb", "ChromaDB vector database"),
        ("PIL", "Pillow for image processing"),
        ("fitz", "PyMuPDF for PDF processing"),
        ("docx", "python-docx for DOCX processing"),
        ("pandas", "Pandas for data processing"),
        ("numpy", "NumPy for numerical operations"),
    ]
    
    missing_deps = []
    
    for module, description in dependencies:
        try:
            __import__(module)
            print(f"✅ {module}: {description}")
        except ImportError:
            print(f"❌ {module}: {description} - MISSING")
            missing_deps.append(module)
    
    if missing_deps:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing_deps)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n✅ All key dependencies available!")
    return True

def cleanup_test_files():
    """Clean up test files created during testing."""
    print("\n🧹 Cleaning up test files...")
    
    test_dirs = ["test_chroma", "test_extracted", "test_chroma_db"]
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            import shutil
            try:
                shutil.rmtree(test_dir)
                print(f"✅ Removed {test_dir}")
            except Exception as e:
                print(f"⚠️  Could not remove {test_dir}: {e}")

def main():
    """Run all tests."""
    print("🔬 Multimodal RAG System - Test Suite")
    print("=" * 50)
    
    tests = [
        ("Dependency Check", check_dependencies),
        ("Import Test", test_imports),
        ("System Initialization", test_system_initialization),
        ("Document Processor", test_document_processor),
        ("Embeddings", test_embeddings),
        ("Vector Database", test_vector_db),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    # Cleanup
    cleanup_test_files()
    
    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Start Ollama: ollama serve")
        print("2. Pull a model: ollama pull llama3.2:3b")
        print("3. Run the web interface: python multimodal_app.py")
        print("4. Or use programmatically: python example_usage.py")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        print("Common issues:")
        print("- Missing dependencies: pip install -r requirements.txt")
        print("- Ollama not installed: Install from https://ollama.ai/")
        print("- Model files not downloaded (will download on first use)")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 