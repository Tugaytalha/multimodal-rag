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

class JinaMultimodalEmbeddings:
    """
    Jina multimodal embedding model that supports both text and image embeddings
    using only on-premises API endpoints.
    """
    
    def __init__(self, 
                 model_name: str = "jinaai/jina-embeddings-v4",
                 api_base_url: Optional[str] = None,
                 force_local: bool = False):
        """
        Initialize Jina multimodal embeddings for on-premises use only.
        
        Args:
            model_name: Jina model name
            api_base_url: On-premises API base URL (e.g., "http://10.144.100.204:38044")
            force_local: Force using local model instead of API
        """
        self.model_name = model_name
        
        # Determine API configuration - only on-premises
        if api_base_url and not force_local:
            # On-premises API endpoint
            self.api_base_url = api_base_url.rstrip('/')
            self.use_api = True
            self.text_embed_url = f"{self.api_base_url}/embed/text"
            self.image_embed_url = f"{self.api_base_url}/embed/image"
            logger.info(f"Using on-premises API at {self.api_base_url}")
        else:
            # Local CLIP model fallback
            logger.info("Using local CLIP model for embeddings")
            self.use_api = False
            self._init_local_clip()
    
    def _init_local_clip(self):
        """Initialize local CLIP model as fallback."""
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            # Use a reliable local CLIP model
            self.clip_model = SentenceTransformer("openai/clip-vit-base-patch32")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.clip_model.to(self.device)
            logger.info(f"Local CLIP model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load local CLIP model: {e}")
            raise ImportError("Failed to initialize local CLIP model. Please install sentence-transformers.")
    
    def _embed_text_with_api(self, texts: List[str]) -> List[List[float]]:
        """Embed text using on-premises API."""
        if not texts:
            return []
            
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "texts": texts
        }
        
        try:
            response = requests.post(self.text_embed_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            # Return text embeddings based on on-premises API format
            return result.get("embeddings", [])
        except requests.exceptions.RequestException as e:
            logger.error(f"On-premises text embedding API failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response content: {e.response.text}")
            raise
    
    def _embed_images_with_api(self, image_paths: List[str]) -> List[List[float]]:
        """Embed images using on-premises API."""
        if not image_paths:
            return []
            
        # Convert images to base64 for API
        images_data = []
        for img_path in image_paths:
            try:
                with open(img_path, "rb") as img_file:
                    img_bytes = img_file.read()
                    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                    images_data.append(img_b64)
            except Exception as e:
                logger.error(f"Failed to encode image {img_path}: {e}")
                raise
        
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "images": images_data
        }
        
        try:
            response = requests.post(self.image_embed_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            # Return image embeddings based on on-premises API format
            return result.get("embeddings", [])
        except requests.exceptions.RequestException as e:
            logger.error(f"On-premises image embedding API failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response content: {e.response.text}")
            raise
    
    def _embed_with_clip(self, texts: List[str] = None, image_paths: List[str] = None) -> List[List[float]]:
        """Embed using local CLIP model."""
        if not hasattr(self, 'clip_model'):
            raise RuntimeError("Local CLIP model not initialized")
            
        embeddings = []
        
        try:
            if texts:
                # Encode texts
                text_embeddings = self.clip_model.encode(
                    texts,
                    convert_to_tensor=True,
                    device=self.device
                )
                embeddings.extend(text_embeddings.cpu().numpy().tolist())
            
            if image_paths:
                # Load and encode images
                from PIL import Image
                for img_path in image_paths:
                    try:
                        image = Image.open(img_path).convert('RGB')
                        img_embedding = self.clip_model.encode(
                            image,
                            convert_to_tensor=True,
                            device=self.device
                        )
                        embeddings.append(img_embedding.cpu().numpy().tolist())
                    except Exception as e:
                        logger.error(f"Failed to process image {img_path}: {e}")
                        # Add zero embedding as placeholder
                        embeddings.append([0.0] * 512)  # Default CLIP embedding size
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Local CLIP embedding failed: {e}")
            raise
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed text using configured method."""
        if not texts:
            return []
            
        if self.use_api:
            return self._embed_text_with_api(texts)
        else:
            return self._embed_with_clip(texts=texts)
    
    def embed_images(self, image_paths: List[str]) -> List[List[float]]:
        """Embed images using configured method."""
        if not image_paths:
            return []
            
        if self.use_api:
            return self._embed_images_with_api(image_paths)
        else:
            return self._embed_with_clip(image_paths=image_paths)
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a query text."""
        embeddings = self.embed_texts([query])
        return embeddings[0] if embeddings else []

class MultimodalEmbeddingManager:
    """
    Manager for different embedding models (text-only and multimodal).
    Handles only on-premises APIs and local models.
    """
    
    def __init__(self,
                 text_embedding_model: str = "jinaai/jina-embeddings-v3",
                 multimodal_embedding_model: str = "jinaai/jina-embeddings-v4",
                 jina_api_base_url: Optional[str] = None,
                 force_local_embeddings: bool = False):
        """
        Initialize embedding manager with on-premises configuration only.
        
        Args:
            text_embedding_model: Model for text embeddings
            multimodal_embedding_model: Model for multimodal embeddings  
            jina_api_base_url: On-premises API base URL
            force_local_embeddings: Force using local models instead of API
        """
        self.text_embedding_model = text_embedding_model
        self.multimodal_embedding_model = multimodal_embedding_model
        self.jina_api_base_url = jina_api_base_url
        self.force_local_embeddings = force_local_embeddings
        
        logger.info(f"Initializing embedding manager...")
        logger.info(f"Text model: {text_embedding_model}")
        logger.info(f"Multimodal model: {multimodal_embedding_model}")
        logger.info(f"On-premises API: {jina_api_base_url if jina_api_base_url else 'None (using local)'}")
        
        # Initialize text embeddings
        self._init_text_embeddings()
        
        # Initialize multimodal embeddings
        self._init_multimodal_embeddings()
        
        logger.info("Embedding manager initialized successfully")
    
    def _init_text_embeddings(self):
        """Initialize text embedding model."""
        try:
            if "jina" in self.text_embedding_model.lower() and self.jina_api_base_url and not self.force_local_embeddings:
                logger.info("Using on-premises API for text embeddings")
                
                # Create custom embeddings wrapper for ChromaDB compatibility
                jina_text_embeddings = JinaMultimodalEmbeddings(
                    model_name=self.text_embedding_model,
                    api_base_url=self.jina_api_base_url,
                    force_local=False
                )
                
                # Wrap for ChromaDB compatibility
                self.text_embeddings = CustomMultimodalEmbeddings(jina_text_embeddings)
                logger.info("Custom multimodal embeddings wrapper created for text")
            else:
                # Use HuggingFace embeddings for text
                logger.info(f"Using HuggingFace embeddings for text: {self.text_embedding_model}")
                self.text_embeddings = HuggingFaceEmbeddings(
                    model_name=self.text_embedding_model,
                    model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
                )
                
        except Exception as e:
            logger.error(f"Failed to initialize text embeddings: {e}")
            logger.info("Falling back to basic text embeddings")
            # Fallback to a simple model
            try:
                self.text_embeddings = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'}
                )
            except Exception as fallback_error:
                logger.error(f"Fallback text embeddings also failed: {fallback_error}")
                raise
    
    def _init_multimodal_embeddings(self):
        """Initialize multimodal embedding model."""
        try:
            # Always use Jina for multimodal embeddings (on-premises or local)
            self.multimodal_embeddings = JinaMultimodalEmbeddings(
                model_name=self.multimodal_embedding_model,
                api_base_url=self.jina_api_base_url,
                force_local=self.force_local_embeddings
            )
            logger.info("Multimodal embeddings initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize multimodal embeddings: {e}")
            raise
    
    def get_text_embeddings(self):
        """Get text embedding model."""
        return self.text_embeddings
    
    def get_multimodal_embeddings(self):
        """Get multimodal embedding model."""
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