import os
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import shutil

# Langchain and ChromaDB
from langchain_chroma import Chroma
from langchain.schema.document import Document

# Custom modules
from multimodal_embeddings import (
    MultimodalEmbeddingManager, 
    CustomMultimodalEmbeddings, 
    PageImageEmbeddings
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultimodalVectorDatabase:
    """
    Vector database manager for handling multiple collections with different embedding strategies.
    """
    
    def __init__(self, 
                 chroma_path: str = "multimodal_chroma",
                 text_embedding_model: str = "jinaai/jina-embeddings-v3",
                 multimodal_embedding_model: str = "jinaai/jina-embeddings-v4",
                 jina_api_base_url: Optional[str] = None,
                 force_local_embeddings: bool = False):
        """
        Initialize multimodal vector database with on-premises configuration only.
        
        Args:
            chroma_path: Path to ChromaDB storage
            text_embedding_model: Model for text embeddings
            multimodal_embedding_model: Model for multimodal embeddings
            jina_api_base_url: On-premises API base URL for embeddings
            force_local_embeddings: Force using local models instead of API
        """
        self.chroma_path = chroma_path
        self.text_embedding_model = text_embedding_model
        self.multimodal_embedding_model = multimodal_embedding_model
        self.jina_api_base_url = jina_api_base_url
        self.force_local_embeddings = force_local_embeddings
        
        # Initialize embedding manager with on-premises configuration
        self.embedding_manager = MultimodalEmbeddingManager(
            text_embedding_model=text_embedding_model,
            multimodal_embedding_model=multimodal_embedding_model,
            jina_api_base_url=jina_api_base_url,
            force_local_embeddings=force_local_embeddings
        )
        
        # Initialize collections
        self.text_collection = None
        self.page_collection = None
        
        logger.info(f"Vector database initialized with path: {chroma_path}")
        if jina_api_base_url:
            logger.info(f"Using custom API endpoint: {jina_api_base_url}")
    
    def _init_collections(self):
        """Initialize ChromaDB collections."""
        try:
            # Text content collection (text, image descriptions, table descriptions)
            self.text_collection = Chroma(
                persist_directory=self.text_collection_path,
                embedding_function=self.embedding_manager.get_text_embedding_function(),
                collection_name="multimodal_content"
            )
            
            # Page images collection (with multimodal embeddings)
            page_embedding_function = PageImageEmbeddings(
                self.embedding_manager.get_multimodal_embedding_function()
            )
            
            self.page_collection = Chroma(
                persist_directory=self.page_collection_path,
                embedding_function=page_embedding_function,
                collection_name="page_images"
            )
            
            logger.info("Collections initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing collections: {e}")
            raise
    
    def clear_database(self):
        """Clear all collections in the database."""
        try:
            logger.info("🗑️ Clearing multimodal database")
            
            # Remove directories
            if os.path.exists(self.chroma_path):
                shutil.rmtree(self.chroma_path)
                logger.info("✅ Database cleared")
            
            # Reset collections
            self.text_collection = None
            self.page_collection = None
            
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collections."""
        stats = {
            "text_content": {"count": 0, "exists": False},
            "page_images": {"count": 0, "exists": False}
        }
        
        try:
            if self.text_collection:
                text_data = self.text_collection.get()
                stats["text_content"]["count"] = len(text_data["ids"])
                stats["text_content"]["exists"] = True
            
            if self.page_collection:
                page_data = self.page_collection.get()
                stats["page_images"]["count"] = len(page_data["ids"])
                stats["page_images"]["exists"] = True
                
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
        
        return stats
    
    def _filter_metadata(self, documents: List[Document]) -> List[Document]:
        """Filter out complex metadata types that ChromaDB can't handle."""
        for doc in documents:
            # Create a new metadata dict with only simple types
            filtered_metadata = {}
            for key, value in doc.metadata.items():
                # Only keep str, int, float, bool, or None values
                if isinstance(value, (str, int, float, bool, type(None))):
                    filtered_metadata[key] = value
                elif isinstance(value, (list, tuple)):
                    # Convert lists/tuples to strings
                    filtered_metadata[key] = str(value)
                else:
                    # Convert other types to strings
                    filtered_metadata[key] = str(value)
            doc.metadata = filtered_metadata
        return documents

    def add_documents(self, documents: List[Document]) -> Dict[str, int]:
        """
        Add documents to appropriate collections based on their type.
        
        Args:
            documents: List of documents to add
            
        Returns:
            Dictionary with count of documents added to each collection
        """
        if not self.text_collection or not self.page_collection:
            self._init_collections()
        
        # Separate documents by collection
        collection_data = self.embedding_manager.process_documents_for_collections(documents)
        
        stats = {"text_content": 0, "page_images": 0}
        
        # Add text content documents
        text_docs = collection_data["text_content"]["documents"]
        if text_docs:
            try:
                # Filter metadata before processing
                text_docs = self._filter_metadata(text_docs)
                
                # Calculate unique IDs for text documents
                text_docs_with_ids = self._calculate_chunk_ids(text_docs, "text")
                existing_text_ids = set(self.text_collection.get()["ids"])
                
                new_text_docs = [
                    doc for doc in text_docs_with_ids 
                    if doc.metadata["id"] not in existing_text_ids
                ]
                
                if new_text_docs:
                    new_text_ids = [doc.metadata["id"] for doc in new_text_docs]
                    self.text_collection.add_documents(new_text_docs, ids=new_text_ids)
                    stats["text_content"] = len(new_text_docs)
                    logger.info(f"Added {len(new_text_docs)} text content documents")
                
            except Exception as e:
                logger.error(f"Error adding text documents: {e}")
        
        # Add page image documents
        page_docs = collection_data["page_images"]["documents"]
        if page_docs:
            try:
                # Filter metadata before processing
                page_docs = self._filter_metadata(page_docs)
                
                # Calculate unique IDs for page documents
                page_docs_with_ids = self._calculate_chunk_ids(page_docs, "page")
                existing_page_ids = set(self.page_collection.get()["ids"])
                
                new_page_docs = [
                    doc for doc in page_docs_with_ids 
                    if doc.metadata["id"] not in existing_page_ids
                ]
                
                if new_page_docs:
                    # For page images, we need to embed the image paths directly
                    page_contents = []
                    for doc in new_page_docs:
                        if "image_path" in doc.metadata and os.path.exists(doc.metadata["image_path"]):
                            page_contents.append(doc.metadata["image_path"])
                        else:
                            page_contents.append(doc.page_content)
                    
                    new_page_ids = [doc.metadata["id"] for doc in new_page_docs]
                    
                    # Update page content to be image paths for embedding
                    for i, doc in enumerate(new_page_docs):
                        doc.page_content = page_contents[i]
                    
                    self.page_collection.add_documents(new_page_docs, ids=new_page_ids)
                    stats["page_images"] = len(new_page_docs)
                    logger.info(f"Added {len(new_page_docs)} page image documents")
                
            except Exception as e:
                logger.error(f"Error adding page documents: {e}")
        
        return stats
    
    def _calculate_chunk_ids(self, documents: List[Document], collection_type: str) -> List[Document]:
        """Calculate unique IDs for documents."""
        for doc in documents:
            if "id" not in doc.metadata:
                # Generate ID based on content hash and metadata
                import hashlib
                content_str = f"{doc.page_content}_{doc.metadata.get('source', '')}"
                content_hash = hashlib.md5(content_str.encode()).hexdigest()[:8]
                doc.metadata["id"] = f"{collection_type}_{content_hash}"
        
        return documents
    
    def hybrid_search(self, 
                     query: str, 
                     k_text: int = 8, 
                     k_pages: int = 2,
                     score_threshold: float = 0.5) -> List[Document]:
        """
        Perform hybrid search across both text content and page image collections.
        
        Args:
            query: Search query
            k_text: Number of text content results to retrieve
            k_pages: Number of page image results to retrieve
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of retrieved documents with scores
        """
        if not self.text_collection or not self.page_collection:
            self._init_collections()
        
        all_results = []
        
        # Search text content collection
        try:
            text_results = self.text_collection.similarity_search_with_score(
                query, k=k_text
            )
            
            for doc, score in text_results:
                if score >= score_threshold:
                    doc.metadata["score"] = score
                    doc.metadata["collection"] = "text_content"
                    all_results.append(doc)
                    
        except Exception as e:
            logger.error(f"Error searching text collection: {e}")
        
        # Search page image collection
        try:
            page_results = self.page_collection.similarity_search_with_score(
                query, k=k_pages
            )
            
            for doc, score in page_results:
                if score >= score_threshold:
                    doc.metadata["score"] = score
                    doc.metadata["collection"] = "page_images"
                    all_results.append(doc)
                    
        except Exception as e:
            logger.error(f"Error searching page collection: {e}")
        
        # Sort by score (lower is better for similarity search)
        all_results.sort(key=lambda x: x.metadata.get("score", float('inf')))
        
        return all_results
    
    def search_by_type(self, 
                      query: str, 
                      search_type: str = "both", 
                      k: int = 10) -> List[Document]:
        """
        Search specific collection type.
        
        Args:
            query: Search query
            search_type: "text", "pages", or "both"
            k: Number of results to retrieve
            
        Returns:
            List of retrieved documents
        """
        if not self.text_collection or not self.page_collection:
            self._init_collections()
        
        results = []
        
        if search_type in ["text", "both"]:
            try:
                text_results = self.text_collection.similarity_search_with_score(query, k=k)
                for doc, score in text_results:
                    doc.metadata["score"] = score
                    doc.metadata["collection"] = "text_content"
                    results.append(doc)
            except Exception as e:
                logger.error(f"Error searching text collection: {e}")
        
        if search_type in ["pages", "both"]:
            try:
                page_results = self.page_collection.similarity_search_with_score(query, k=k)
                for doc, score in page_results:
                    doc.metadata["score"] = score
                    doc.metadata["collection"] = "page_images"
                    results.append(doc)
            except Exception as e:
                logger.error(f"Error searching page collection: {e}")
        
        # Sort by score
        results.sort(key=lambda x: x.metadata.get("score", float('inf')))
        
        return results[:k]

class MultimodalRAGRetriever:
    """
    High-level retriever for multimodal RAG system.
    """
    
    def __init__(self, vector_db: MultimodalVectorDatabase):
        """
        Initialize the retriever.
        
        Args:
            vector_db: Multimodal vector database instance
        """
        self.vector_db = vector_db
    
    def retrieve(self, 
                query: str,
                retrieval_strategy: str = "hybrid",
                k_text: int = 8,
                k_pages: int = 2,
                **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            retrieval_strategy: "hybrid", "text_only", "pages_only"
            k_text: Number of text results
            k_pages: Number of page results
            
        Returns:
            List of retrieved documents with metadata
        """
        if retrieval_strategy == "hybrid":
            documents = self.vector_db.hybrid_search(query, k_text, k_pages)
        elif retrieval_strategy == "text_only":
            documents = self.vector_db.search_by_type(query, "text", k_text)
        elif retrieval_strategy == "pages_only":
            documents = self.vector_db.search_by_type(query, "pages", k_pages)
        else:
            raise ValueError(f"Invalid retrieval strategy: {retrieval_strategy}")
        
        # Format results
        results = []
        for doc in documents:
            result = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": doc.metadata.get("score", 0.0),
                "collection": doc.metadata.get("collection", "unknown"),
                "element_type": doc.metadata.get("element_type", "text"),
            }
            
            # Add special handling for different content types
            if doc.metadata.get("element_type") in ["image", "graph"]:
                result["image_path"] = doc.metadata.get("image_path")
                result["description"] = doc.metadata.get("description", doc.page_content)
                result["retrieve_content"] = doc.metadata.get("image_path")
                
            elif doc.metadata.get("element_type") == "table":
                result["table_html"] = doc.metadata.get("table_html")
                result["description"] = doc.metadata.get("description", doc.page_content)
                result["retrieve_content"] = doc.metadata.get("table_html")
                
            elif doc.metadata.get("element_type") == "page_image":
                result["image_path"] = doc.metadata.get("image_path")
                result["page"] = doc.metadata.get("page")
                result["retrieve_content"] = doc.metadata.get("image_path")
            
            results.append(result)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return self.vector_db.get_collection_stats()

if __name__ == "__main__":
    # Example usage
    vector_db = MultimodalVectorDatabase()
    retriever = MultimodalRAGRetriever(vector_db)
    
    # Get stats
    stats = retriever.get_stats()
    print(f"Database stats: {stats}")
    
    # Example search (would need actual documents in database)
    try:
        results = retriever.retrieve("financial reports", retrieval_strategy="hybrid")
        print(f"Found {len(results)} results")
        for result in results[:3]:  # Show first 3 results
            print(f"- {result['element_type']}: {result['content'][:100]}...")
    except Exception as e:
        print(f"Search failed: {e}") 