import os
import logging
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
import base64
import io

# Langchain embeddings
from langchain_core.embeddings import Embeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.schema.document import Document

# Jina multimodal embeddings
import requests
import torch
from transformers import CLIPModel, CLIPProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultimodalEmbeddingBase(ABC):
    """Base class for multimodal embeddings."""
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed text documents."""
        pass
    
    @abstractmethod
    def embed_images(self, image_paths: List[str]) -> List[List[float]]:
        """Embed images."""
        pass
    
    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Embed a query text."""
        pass

class JinaMultimodalEmbeddings(MultimodalEmbeddingBase):
    """
    Jina multimodal embeddings (jina-embeddings-v4) for handling both text and images.
    """
    
    def __init__(self, 
                 model_name: str = "jinaai/jina-embeddings-v4",
                 api_key: Optional[str] = None,
                 base_url: str = "https://api.jina.ai/v1/embeddings"):
        """
        Initialize Jina multimodal embeddings.
        
        Args:
            model_name: Jina model name
            api_key: Jina API key (if using API)
            base_url: Jina API base URL
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        self.base_url = base_url
        
        # Try to use local model if API key is not available
        if not self.api_key:
            logger.warning("No Jina API key found. Falling back to local CLIP model.")
            self.use_api = False
            self._init_local_clip()
        else:
            self.use_api = True
            logger.info("Using Jina API for multimodal embeddings")
    
    def _init_local_clip(self):
        """Initialize local CLIP model as fallback."""
        try:
            self.use_api = False
            self.clip_model = CLIPModel.from_pretrained("jinaai/jina-embeddings-v4")
            self.clip_processor = CLIPProcessor.from_pretrained("jinaai/jina-embeddings-v4")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model.to(self.device)
            logger.info(f"Local CLIP model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load local CLIP model: {e}")
            raise
    
    def _image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _embed_with_api(self, inputs: List[Dict[str, Any]]) -> List[List[float]]:
        """Embed using Jina API."""
        if not inputs:
            return []
            
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model_name,
            "input": inputs,
            "encoding_format": "float"
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return [item["embedding"] for item in result["data"]]
        except requests.exceptions.RequestException as e:
            logger.error(f"Jina API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response content: {e.response.text}")
            # Fall back to local model
            logger.info("Falling back to local CLIP model")
            if not hasattr(self, 'clip_model'):
                self._init_local_clip()
            self.use_api = False
            # Retry with local model
            if any("image" in inp.get("type", "") for inp in inputs):
                image_paths = [inp.get("image_path", "") for inp in inputs if inp.get("type") == "image"]
                return self._embed_with_clip(image_paths=image_paths)
            else:
                texts = [inp.get("text", "") for inp in inputs if inp.get("type") == "text"]
                return self._embed_with_clip(texts=texts)
    
    def _embed_with_clip(self, texts: List[str] = None, image_paths: List[str] = None) -> List[List[float]]:
        """Embed using local CLIP model."""
        embeddings = []
        
        if texts:
            inputs = self.clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                text_embeddings = self.clip_model.get_text_features(**inputs)
                text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
                embeddings.extend(text_embeddings.cpu().numpy().tolist())
        
        if image_paths:
            for image_path in image_paths:
                try:
                    image = Image.open(image_path).convert("RGB")
                    inputs = self.clip_processor(images=image, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        image_embedding = self.clip_model.get_image_features(**inputs)
                        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
                        embeddings.append(image_embedding.cpu().numpy().tolist()[0])
                        
                except Exception as e:
                    logger.error(f"Error processing image {image_path}: {e}")
                    # Return zero embedding as fallback
                    embeddings.append([0.0] * 512)  # CLIP base has 512 dimensions
        
        return embeddings
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed text documents."""
        if self.use_api:
            inputs = [{"type": "text", "text": text} for text in texts]
            return self._embed_with_api(inputs)
        else:
            return self._embed_with_clip(texts=texts)
    
    def embed_images(self, image_paths: List[str]) -> List[List[float]]:
        """Embed images."""
        if self.use_api:
            inputs = []
            for image_path in image_paths:
                try:
                    image_b64 = self._image_to_base64(image_path)
                    inputs.append({"type": "image", "image": image_b64})
                except Exception as e:
                    logger.error(f"Error processing image {image_path}: {e}")
                    # Skip this image
                    continue
            
            if inputs:
                return self._embed_with_api(inputs)
            else:
                return []
        else:
            return self._embed_with_clip(image_paths=image_paths)
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a query text."""
        return self.embed_texts([query])[0]

class MultimodalEmbeddingManager:
    """
    Manager for handling different types of embeddings based on content type.
    """
    
    def __init__(self,
                 text_embedding_model: str = "jinaai/jina-embeddings-v3",
                 multimodal_embedding_model: str = "jinaai/jina-embeddings-v4",
                 jina_api_key: Optional[str] = None):
        """
        Initialize the multimodal embedding manager.
        
        Args:
            text_embedding_model: Model for text embeddings
            multimodal_embedding_model: Model for multimodal embeddings
            jina_api_key: Jina API key for multimodal embeddings
        """
        self.text_embedding_model = text_embedding_model
        self.multimodal_embedding_model = multimodal_embedding_model
        
        # Initialize text embeddings
        self.text_embeddings = HuggingFaceEmbeddings(
            model_name=text_embedding_model,
            encode_kwargs={'normalize_embeddings': True},
            model_kwargs={'trust_remote_code': True}
        )
        
        # Initialize multimodal embeddings
        self.multimodal_embeddings = JinaMultimodalEmbeddings(
            model_name=multimodal_embedding_model,
            api_key=jina_api_key
        )
        
        logger.info(f"Embedding manager initialized with text model: {text_embedding_model}, multimodal model: {multimodal_embedding_model}")
    
    def get_text_embedding_function(self):
        """Get the text embedding function."""
        return self.text_embeddings
    
    def get_multimodal_embedding_function(self):
        """Get the multimodal embedding function."""
        return self.multimodal_embeddings
    
    def embed_documents_by_type(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """
        Separate documents by type and embed them with appropriate models.
        
        Args:
            documents: List of documents to embed
            
        Returns:
            Dictionary with document types as keys and embedded documents as values
        """
        # Separate documents by type
        text_docs = []
        image_docs = []
        table_docs = []
        page_image_docs = []
        
        for doc in documents:
            element_type = doc.metadata.get("element_type", "text")
            
            if element_type == "text":
                text_docs.append(doc)
            elif element_type in ["image", "graph"]:
                image_docs.append(doc)
            elif element_type == "table":
                table_docs.append(doc)
            elif element_type == "page_image":
                page_image_docs.append(doc)
        
        # Embed text documents (including descriptions from images and tables)
        text_and_description_docs = text_docs + image_docs + table_docs
        
        return {
            "text_content": text_and_description_docs,  # Text + image/table descriptions
            "page_images": page_image_docs               # Page images for multimodal embedding
        }
    
    def process_documents_for_collections(self, documents: List[Document]) -> Dict[str, Dict[str, Any]]:
        """
        Process documents and prepare them for different collections.
        
        Args:
            documents: List of documents to process
            
        Returns:
            Dictionary with collection data
        """
        # Separate documents by type
        doc_types = self.embed_documents_by_type(documents)
        
        # Prepare text content collection (regular text embeddings)
        text_collection_data = {
            "documents": doc_types["text_content"],
            "embedding_function": self.text_embeddings,
            "collection_name": "multimodal_content"
        }
        
        # Prepare page images collection (multimodal embeddings)
        page_collection_data = {
            "documents": doc_types["page_images"],
            "embedding_function": self.multimodal_embeddings,
            "collection_name": "page_images"
        }
        
        return {
            "text_content": text_collection_data,
            "page_images": page_collection_data
        }

class CustomMultimodalEmbeddings(Embeddings):
    """
    Custom embeddings class that wraps multimodal embeddings for ChromaDB compatibility.
    """
    
    def __init__(self, multimodal_embeddings: JinaMultimodalEmbeddings):
        self.multimodal_embeddings = multimodal_embeddings
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        return self.multimodal_embeddings.embed_texts(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query string."""
        return self.multimodal_embeddings.embed_query(text)

class PageImageEmbeddings(Embeddings):
    """
    Specialized embeddings for page images that can handle both text queries and image content.
    """
    
    def __init__(self, multimodal_embeddings: JinaMultimodalEmbeddings):
        self.multimodal_embeddings = multimodal_embeddings
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed page image documents. The 'texts' here are actually image paths or image descriptions.
        """
        # Check if inputs are image paths or text descriptions
        embeddings = []
        
        for text in texts:
            # Try to determine if this is an image path or text description
            if os.path.exists(text) and any(text.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']):
                # This is an image path
                try:
                    img_embeddings = self.multimodal_embeddings.embed_images([text])
                    embeddings.extend(img_embeddings)
                except Exception as e:
                    logger.error(f"Error embedding image {text}: {e}")
                    # Fallback to text embedding of the path
                    text_embeddings = self.multimodal_embeddings.embed_texts([f"Image: {text}"])
                    embeddings.extend(text_embeddings)
            else:
                # This is text description - embed as text
                try:
                    text_embeddings = self.multimodal_embeddings.embed_texts([text])
                    embeddings.extend(text_embeddings)
                except Exception as e:
                    logger.error(f"Error embedding text {text}: {e}")
                    # Return zero embedding as fallback
                    embeddings.append([0.0] * 512)  # Default embedding size
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query string for page image search."""
        return self.multimodal_embeddings.embed_query(text)

if __name__ == "__main__":
    # Example usage
    manager = MultimodalEmbeddingManager()
    
    # Test text embedding
    text_embeddings = manager.text_embeddings.embed_documents(["This is a test document"])
    print(f"Text embedding shape: {np.array(text_embeddings).shape}")
    
    # Test multimodal embedding (if API key is available)
    try:
        multimodal_embeddings = manager.multimodal_embeddings.embed_texts(["This is a test for multimodal"])
        print(f"Multimodal text embedding shape: {np.array(multimodal_embeddings).shape}")
    except Exception as e:
        print(f"Multimodal embedding test failed: {e}") 