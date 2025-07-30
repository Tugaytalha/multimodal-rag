#!/usr/bin/env python3
"""
Dependency checker and fixer for the Multimodal RAG System
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def check_and_install_dependencies():
    """Check for missing dependencies and install them."""
    
    print("🔧 Checking and installing missing dependencies...")
    
    required_packages = [
        "python-docx",
        "PyMuPDF", 
        "chromadb",
        "langchain-community",
        "langchain-chroma",
        "langchain-huggingface",
        "beautifulsoup4",
        "html2text",
        "torch",
        "torchvision",
        "transformers",
        "sentence-transformers",
        "requests",
        "pandas",
        "numpy",
        "Pillow",
        "gradio"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == "python-docx":
                import docx
            elif package == "PyMuPDF":
                import fitz
            elif package == "chromadb":
                import chromadb
            elif package == "langchain-community":
                import langchain_community
            elif package == "langchain-chroma":
                import langchain_chroma
            elif package == "langchain-huggingface":
                import langchain_huggingface
            elif package == "beautifulsoup4":
                import bs4
            elif package == "html2text":
                import html2text
            elif package == "torch":
                import torch
            elif package == "torchvision":
                import torchvision
            elif package == "transformers":
                import transformers
            elif package == "sentence-transformers":
                import sentence_transformers
            elif package == "requests":
                import requests
            elif package == "pandas":
                import pandas
            elif package == "numpy":
                import numpy
            elif package == "Pillow":
                import PIL
            elif package == "gradio":
                import gradio
            
            print(f"✅ {package}")
            
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Installing {len(missing_packages)} missing packages...")
        
        for package in missing_packages:
            print(f"Installing {package}...")
            if install_package(package):
                print(f"✅ {package} installed successfully")
            else:
                print(f"❌ Failed to install {package}")
    
    else:
        print("\n🎉 All dependencies are already installed!")
    
    return len(missing_packages) == 0

def test_imports():
    """Test if the multimodal system can be imported."""
    print("\n🧪 Testing multimodal system imports...")
    
    try:
        print("Testing multimodal_document_processor...")
        import multimodal_document_processor
        print("✅ multimodal_document_processor imported")
        
        print("Testing multimodal_embeddings...")
        import multimodal_embeddings
        print("✅ multimodal_embeddings imported")
        
        print("Testing multimodal_vector_db...")
        import multimodal_vector_db
        print("✅ multimodal_vector_db imported")
        
        print("Testing multimodal_rag_system...")
        import multimodal_rag_system
        print("✅ multimodal_rag_system imported")
        
        print("\n🎉 All core modules imported successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

if __name__ == "__main__":
    print("🔬 Multimodal RAG System - Dependency Checker")
    print("=" * 50)
    
    # Check and install dependencies
    deps_ok = check_and_install_dependencies()
    
    # Test imports
    imports_ok = test_imports()
    
    print("\n" + "=" * 50)
    if deps_ok and imports_ok:
        print("🎉 System is ready!")
        print("\nNext steps:")
        print("1. Run: python multimodal_app.py")
        print("2. Open: http://localhost:7860")
        print("3. Upload documents and start querying!")
    else:
        print("⚠️ Some issues remain. Check the errors above.")
        print("\nManual installation commands:")
        print("pip install python-docx PyMuPDF chromadb langchain-community")
        print("pip install langchain-chroma langchain-huggingface torch transformers")
        print("pip install sentence-transformers gradio beautifulsoup4 html2text") 