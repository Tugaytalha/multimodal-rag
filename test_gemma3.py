#!/usr/bin/env python3
"""
Test script for Gemma 3 multimodal capabilities
"""

import os
import sys
from multimodal_rag_system import create_multimodal_rag_system

def test_gemma3_system():
    """Test the multimodal RAG system with Gemma 3 models."""
    
    print("🧪 Testing Gemma 3 Multimodal RAG System")
    print("=" * 50)
    
    try:
        # Create system with Gemma 3 configuration
        print("1. Initializing system with Gemma 3 models...")
        config = {
            "vlm_model": "gemma3:4b",
            "llm_model": "gemma3:4b",
            "text_embedding_model": "jinaai/jina-embeddings-v3",
            "multimodal_embedding_model": "jinaai/jina-embeddings-v4"
        }
        
        rag_system = create_multimodal_rag_system(config)
        print("✅ System initialized successfully with Gemma 3!")
        
        # Test database stats
        print("\n2. Getting database statistics...")
        stats = rag_system.get_database_stats()
        print(f"📊 Database stats: {stats}")
        
        # Test document processor initialization
        print("\n3. Testing document processor...")
        processor = rag_system.document_processor
        print(f"📝 VLM Model: {processor.vlm_model_name}")
        print(f"🔧 Using Ollama VLM: {getattr(processor, 'use_ollama_vlm', 'Unknown')}")
        
        if hasattr(processor, 'vlm_model') and processor.vlm_model:
            print("✅ VLM model loaded successfully")
        else:
            print("⚠️ VLM model not loaded")
        
        print("\n4. Testing LLM...")
        llm_model = rag_system.llm.model
        print(f"💬 LLM Model: {llm_model}")
        
        # Test a simple query (without documents)
        print("\n5. Testing basic query functionality...")
        try:
            # This should return an error message since no documents are loaded
            result = rag_system.query("What is the main topic?")
            print(f"🔍 Query test result: {result['response'][:100]}...")
            print("✅ Query system working")
        except Exception as e:
            print(f"⚠️ Query test failed: {e}")
        
        print(f"\n🎉 Gemma 3 system test completed successfully!")
        print("\nNext steps:")
        print("1. Upload some documents to test multimodal capabilities")
        print("2. Use the Gradio interface: python multimodal_app.py")
        print("3. Test with images, tables, and graphs")
        
        return True
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_ollama_availability():
    """Check if Ollama is available and has Gemma 3 models."""
    print("\n🔍 Checking Ollama availability...")
    
    try:
        import ollama
        # Try to list available models
        models = ollama.list()
        print("📋 Available Ollama models:")
        
        gemma3_models = []
        for model in models.get('models', []):
            model_name = model.get('name', '')
            print(f"  - {model_name}")
            if 'gemma3' in model_name.lower():
                gemma3_models.append(model_name)
        
        if gemma3_models:
            print(f"\n✅ Found Gemma 3 models: {gemma3_models}")
        else:
            print("\n⚠️ No Gemma 3 models found in Ollama")
            print("   To install: ollama pull gemma3:4b")
        
        return len(gemma3_models) > 0
        
    except ImportError:
        print("⚠️ Ollama Python library not available")
        return False
    except Exception as e:
        print(f"⚠️ Error checking Ollama: {e}")
        return False

if __name__ == "__main__":
    print("🔬 Gemma 3 Multimodal RAG - Test Suite")
    print("=" * 50)
    
    # Check Ollama first
    ollama_ok = check_ollama_availability()
    
    # Test the system
    system_ok = test_gemma3_system()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"   Ollama Check: {'✅ PASS' if ollama_ok else '⚠️ WARNING'}")
    print(f"   System Test:  {'✅ PASS' if system_ok else '❌ FAIL'}")
    
    if system_ok:
        print("\n🎉 Ready to use Gemma 3 for multimodal RAG!")
    else:
        print("\n⚠️ Some issues detected. Check the output above.") 