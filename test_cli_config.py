#!/usr/bin/env python3
"""
Test CLI Configuration - Verify Jina v4 and Ollama setup
"""

import requests
import json
import base64
from PIL import Image, ImageDraw
import io
import os

def test_jina_text_embedding(api_url="http://10.144.100.204:38044"):
    """Test Jina text embeddings like user's code"""
    print("🧪 Testing Jina Text Embeddings...")
    
    test_text = "Merhaba dünya, bu bir test metnidir."
    payload = {
        "model": "jinaai/jina-embeddings-v4",
        "texts": [test_text],
        "task": "retrieval",
        "prompt_name": "query"
    }
    
    try:
        response = requests.post(
            f"{api_url}/embed/text",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Text embedding successful!")
            print(f"  Model: {result['model']}")
            print(f"  Shape: {result['shape']}")
            print(f"  Task: {result['task']}")
            print(f"  First 5 values: {result['embeddings'][0][:5]}")
            return True
        else:
            print(f"✗ Text embedding failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Text embedding error: {e}")
        return False

def test_jina_image_embedding(api_url="http://10.144.100.204:38044"):
    """Test Jina image embeddings like user's code"""
    print("\n🖼️ Testing Jina Image Embeddings...")
    
    # Create test image
    img = Image.new('RGB', (200, 200), color='blue')
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 150, 150], fill='red')
    draw.ellipse([75, 75, 125, 125], fill='yellow')
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    queries = [
        "Bu görüntüde ne var?",
        "Hangi renkler görülüyor?",
        "Geometrik şekiller var mı?"
    ]
    
    payload = {
        "model": "jinaai/jina-embeddings-v4",
        "queries": queries,
        "images_base64": [encoded]
    }
    
    try:
        response = requests.post(
            f"{api_url}/embed/image",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Image embedding successful!")
            print(f"  Model: {result['model']}")
            print(f"  Images: {result['num_images']}")
            print(f"  Queries: {result['num_queries']}")
            print(f"  Image embedding size: {len(result['image_embeddings'][0])}")
            print(f"  Query embedding size: {len(result['query_embeddings'][0])}")
            
            print("\n📊 Similarity scores:")
            for i, (query, score) in enumerate(zip(queries, result['similarity_scores'][0])):
                print(f"  {i+1}. {score:.4f} - '{query}'")
            return True
        else:
            print(f"✗ Image embedding failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Image embedding error: {e}")
        return False

def test_ollama_connection(api_url="http://111.111.11.11.1:11434"):
    """Test Ollama connection"""
    print("\n🤖 Testing Ollama Connection...")
    
    try:
        # Test list models
        response = requests.get(f"{api_url}/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json()
            print("✓ Ollama connected successfully!")
            print(f"  Available models: {len(models.get('models', []))}")
            
            # Check for gemma3:27b
            model_names = [m.get('model', m.get('name', '')) for m in models.get('models', [])]
            if 'gemma3:27b' in model_names:
                print("✓ gemma3:27b model found!")
            else:
                print("⚠️ gemma3:27b model not found")
                print(f"Available models: {model_names[:5]}")  # Show first 5
            
            return True
        else:
            print(f"✗ Ollama connection failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Ollama connection error: {e}")
        return False

def test_ollama_generate(api_url="http://111.111.11.11.1:11434", model="gemma3:27b"):
    """Test Ollama generate endpoint"""
    print(f"\n💬 Testing Ollama Generate with {model}...")
    
    payload = {
        "model": model,
        "prompt": "What is the capital of Turkey?",
        "stream": False
    }
    
    try:
        response = requests.post(
            f"{api_url}/api/generate",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Ollama generate successful!")
            print(f"  Response: {result.get('response', '')[:100]}...")
            return True
        else:
            print(f"✗ Ollama generate failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Ollama generate error: {e}")
        return False

def test_embedding_configuration():
    """Test the new flexible embedding configuration"""
    print("\n🔧 Testing Embedding Configuration...")
    
    try:
        # Test the new MultimodalEmbeddingManager with custom API
        from multimodal_embeddings import MultimodalEmbeddingManager
        
        # Test API configuration
        print("   Testing custom API configuration...")
        manager_api = MultimodalEmbeddingManager(
            text_embedding_model="jinaai/jina-embeddings-v4",
            multimodal_embedding_model="jinaai/jina-embeddings-v4",
            jina_api_base_url="http://10.144.100.204:38044"
        )
        print("   ✓ API configuration created successfully")
        
        # Test local configuration  
        print("   Testing local configuration...")
        manager_local = MultimodalEmbeddingManager(
            text_embedding_model="jinaai/jina-embeddings-v3",
            multimodal_embedding_model="jinaai/jina-embeddings-v4",
            force_local_embeddings=True
        )
        print("   ✓ Local configuration created successfully")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Configuration test failed: {e}")
        return False

def main():
    print("🔬 CLI Configuration Test")
    print("=" * 50)
    
    # Test configurations
    jina_api_url = "http://10.144.100.204:38044"
    ollama_api_url = "http://111.111.11.11.1:11434"
    
    print(f"Jina API: {jina_api_url}")
    print(f"Ollama API: {ollama_api_url}")
    print(f"Model: gemma3:27b")
    print(f"Embeddings: jinaai/jina-embeddings-v4")
    
    # Run tests
    jina_text_ok = test_jina_text_embedding(jina_api_url)
    jina_image_ok = test_jina_image_embedding(jina_api_url)
    ollama_conn_ok = test_ollama_connection(ollama_api_url)
    ollama_gen_ok = test_ollama_generate(ollama_api_url) if ollama_conn_ok else False
    config_ok = test_embedding_configuration()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"   Jina Text Embedding:  {'✓ PASS' if jina_text_ok else '✗ FAIL'}")
    print(f"   Jina Image Embedding: {'✓ PASS' if jina_image_ok else '✗ FAIL'}")
    print(f"   Ollama Connection:    {'✓ PASS' if ollama_conn_ok else '✗ FAIL'}")
    print(f"   Ollama Generate:      {'✓ PASS' if ollama_gen_ok else '✗ FAIL'}")
    print(f"   Embedding Config:     {'✓ PASS' if config_ok else '✗ FAIL'}")
    
    all_ok = jina_text_ok and jina_image_ok and ollama_conn_ok and ollama_gen_ok and config_ok
    
    if all_ok:
        print("\n🎉 All tests passed! CLI is ready to use.")
    else:
        print("\n⚠️ Some tests failed. Check the configuration.")
    
    print("\nTo run CLI: python cli_multimodal_rag.py")

if __name__ == "__main__":
    main() 