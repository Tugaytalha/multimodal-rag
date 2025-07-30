import os
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Langchain components
from langchain.prompts import ChatPromptTemplate
# Try different import options for Ollama
try:
    from langchain_ollama import OllamaLLM as Ollama
except ImportError:
    try:
        from langchain_community.llms import Ollama
    except ImportError:
        try:
            from langchain.llms import Ollama
        except ImportError:
            print("⚠️  Warning: Ollama not found. Please install one of:")
            print("   - pip install langchain-ollama")
            print("   - pip install langchain-community") 
            print("   - Ensure Ollama is available in your langchain installation")
            
            # Create a dummy Ollama class as fallback
            class Ollama:
                def __init__(self, model="gemma3:4b"):
                    self.model = model
                    print(f"⚠️  Using dummy Ollama class. Please install proper Ollama support.")
                
                def invoke(self, prompt):
                    return "Error: Ollama not properly installed. Please install langchain-community or langchain-ollama and ensure Ollama is running."

from langchain.schema.document import Document

# Custom modules
try:
    from multimodal_document_processor import MultimodalDocumentProcessor, ExtractedElement
except ImportError:
    try:
        from ultimodal_document_processor import MultimodalDocumentProcessor, ExtractedElement
    except ImportError:
        print("⚠️ Warning: Could not import document processor. Some features may not work.")
        MultimodalDocumentProcessor = None
        ExtractedElement = None

try:
    from multimodal_vector_db import MultimodalVectorDatabase, MultimodalRAGRetriever
except ImportError:
    try:
        from ultimodal_vector_db import MultimodalVectorDatabase, MultimodalRAGRetriever
    except ImportError:
        print("⚠️ Warning: Could not import vector database. Some features may not work.")
        MultimodalVectorDatabase = None
        MultimodalRAGRetriever = None

try:
    from multimodal_embeddings import MultimodalEmbeddingManager
except ImportError:
    try:
        from ultimodal_embeddings import MultimodalEmbeddingManager
    except ImportError:
        print("⚠️ Warning: Could not import embedding manager. Some features may not work.")
        MultimodalEmbeddingManager = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultimodalRAGSystem:
    """
    Complete multimodal RAG system that handles document processing, embedding, 
    storage, retrieval, and response generation.
    """
    
    def __init__(self,
                 chroma_path: str = "multimodal_chroma",
                 extracted_content_path: str = "extracted_content",
                 text_embedding_model: str = "jinaai/jina-embeddings-v4",
                 multimodal_embedding_model: str = "jinaai/jina-embeddings-v4",
                 vlm_model: str = "gemma3:4b",
                 llm_model: str = "gemma3:4b",
                 jina_api_key: Optional[str] = None):
        """
        Initialize the multimodal RAG system.
        
        Args:
            chroma_path: Path for ChromaDB storage
            extracted_content_path: Path for storing extracted images and pages
            text_embedding_model: Model for text embeddings
            multimodal_embedding_model: Model for multimodal embeddings
            vlm_model: Vision language model for image descriptions
            llm_model: Language model for response generation
            jina_api_key: API key for Jina embeddings
        """
        self.chroma_path = chroma_path
        self.extracted_content_path = extracted_content_path
        self.llm_model = llm_model
        
        # Check if required components are available
        if MultimodalDocumentProcessor is None:
            raise ImportError("MultimodalDocumentProcessor is not available. Please install missing dependencies.")
        if MultimodalVectorDatabase is None:
            raise ImportError("MultimodalVectorDatabase is not available. Please install missing dependencies.")
        if MultimodalEmbeddingManager is None:
            raise ImportError("MultimodalEmbeddingManager is not available. Please install missing dependencies.")
        
        # Initialize components
        self.document_processor = MultimodalDocumentProcessor(
            vlm_model_name=vlm_model
        )
        
        self.vector_db = MultimodalVectorDatabase(
            chroma_path=chroma_path,
            text_embedding_model=text_embedding_model,
            multimodal_embedding_model=multimodal_embedding_model,
            jina_api_key=jina_api_key
        )
        
        self.retriever = MultimodalRAGRetriever(self.vector_db)
        
        # Initialize LLM
        self.llm = Ollama(model=llm_model)
        
        # Prompt template for multimodal responses
        self.prompt_template = ChatPromptTemplate.from_template("""
You are a helpful AI assistant that can analyze and answer questions based on multimodal document content including text, images, tables, and graphs.

Context from retrieved documents:
{context}

Instructions:
- Answer the question based on the provided context
- When referencing images or graphs, describe what you see and how it relates to the question
- When referencing tables, summarize the key information that answers the question
- If the context contains page images, mention that visual information is available
- Be specific about which type of content (text, image, table, graph) you're referencing
- If you cannot answer based on the provided context, say so clearly

Question: {question}

Answer:""")
        
        logger.info("Multimodal RAG system initialized successfully")
    
    def clear_database(self):
        """Clear the vector database."""
        self.vector_db.clear_database()
        logger.info("Database cleared")
    
    def process_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Process multiple documents and add them to the vector database.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            Processing statistics
        """
        total_elements = 0
        total_documents = 0
        processing_stats = {
            "files_processed": 0,
            "total_elements": 0,
            "total_documents": 0,
            "elements_by_type": {},
            "errors": []
        }
        
        for file_path in file_paths:
            try:
                logger.info(f"Processing file: {file_path}")
                start_time = time.time()
                
                # Extract elements from document
                elements = self.document_processor.process_document(
                    file_path, 
                    self.extracted_content_path
                )
                
                # Convert elements to documents
                documents = self.document_processor.elements_to_documents(elements)
                
                # Add documents to vector database
                db_stats = self.vector_db.add_documents(documents)
                
                # Update statistics
                processing_stats["files_processed"] += 1
                processing_stats["total_elements"] += len(elements)
                processing_stats["total_documents"] += len(documents)
                
                # Count elements by type
                for element in elements:
                    element_type = element.element_type
                    if element_type not in processing_stats["elements_by_type"]:
                        processing_stats["elements_by_type"][element_type] = 0
                    processing_stats["elements_by_type"][element_type] += 1
                
                processing_time = time.time() - start_time
                logger.info(f"Processed {file_path} in {processing_time:.2f}s: "
                          f"{len(elements)} elements, {len(documents)} documents")
                
            except Exception as e:
                error_msg = f"Error processing {file_path}: {str(e)}"
                logger.error(error_msg)
                processing_stats["errors"].append(error_msg)
        
        return processing_stats
    
    def query(self,
              question: str,
              retrieval_strategy: str = "hybrid",
              k_text: int = 8,
              k_pages: int = 2,
              include_images: bool = True,
              include_tables: bool = True) -> Dict[str, Any]:
        """
        Query the multimodal RAG system.
        
        Args:
            question: User question
            retrieval_strategy: "hybrid", "text_only", "pages_only"
            k_text: Number of text results to retrieve
            k_pages: Number of page results to retrieve
            include_images: Whether to include image information in response
            include_tables: Whether to include table information in response
            
        Returns:
            Dictionary containing response and metadata
        """
        start_time = time.time()
        
        try:
            # Retrieve relevant documents
            retrieved_docs = self.retriever.retrieve(
                query=question,
                retrieval_strategy=retrieval_strategy,
                k_text=k_text,
                k_pages=k_pages
            )
            
            if not retrieved_docs:
                return {
                    "response": "I couldn't find any relevant information to answer your question.",
                    "sources": [],
                    "retrieved_content": [],
                    "processing_time": time.time() - start_time
                }
            
            # Format context for LLM
            context_parts = []
            sources = set()
            retrieved_content = []
            
            for i, doc in enumerate(retrieved_docs):
                element_type = doc["element_type"]
                source = doc["metadata"].get("source", "Unknown")
                sources.add(source)
                
                if element_type == "text":
                    context_parts.append(f"Text Content {i+1}:\n{doc['content']}\n")
                    retrieved_content.append({
                        "type": "text",
                        "content": doc["content"],
                        "source": source,
                        "score": doc["score"]
                    })
                
                elif element_type in ["image", "graph"] and include_images:
                    description = doc.get("description", doc["content"])
                    image_path = doc.get("image_path", "")
                    context_parts.append(
                        f"{'Graph' if element_type == 'graph' else 'Image'} {i+1} Description:\n"
                        f"{description}\n"
                        f"Source: {source}\n"
                    )
                    retrieved_content.append({
                        "type": element_type,
                        "description": description,
                        "image_path": image_path,
                        "source": source,
                        "score": doc["score"]
                    })
                
                elif element_type == "table" and include_tables:
                    description = doc.get("description", doc["content"])
                    table_html = doc.get("table_html", "")
                    context_parts.append(
                        f"Table {i+1} Description:\n"
                        f"{description}\n"
                        f"Source: {source}\n"
                    )
                    retrieved_content.append({
                        "type": "table",
                        "description": description,
                        "table_html": table_html,
                        "source": source,
                        "score": doc["score"]
                    })
                
                elif element_type == "page_image":
                    page_num = doc["metadata"].get("page", "Unknown")
                    image_path = doc.get("image_path", "")
                    context_parts.append(
                        f"Page {page_num} Image:\n"
                        f"Visual content from page {page_num} of document\n"
                        f"Source: {source}\n"
                    )
                    retrieved_content.append({
                        "type": "page_image",
                        "page": page_num,
                        "image_path": image_path,
                        "source": source,
                        "score": doc["score"]
                    })
            
            # Combine context
            context = "\n".join(context_parts)
            
            # Generate response
            prompt = self.prompt_template.format(context=context, question=question)
            response = self.llm.invoke(prompt)
            
            processing_time = time.time() - start_time
            
            return {
                "response": response,
                "sources": list(sources),
                "retrieved_content": retrieved_content,
                "context": context,
                "processing_time": processing_time,
                "retrieval_stats": {
                    "total_retrieved": len(retrieved_docs),
                    "by_type": self._count_by_type(retrieved_docs),
                    "strategy": retrieval_strategy
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "response": f"Error processing query: {str(e)}",
                "sources": [],
                "retrieved_content": [],
                "processing_time": time.time() - start_time,
                "retrieval_stats": {
                    "total_retrieved": 0,
                    "by_type": {},
                    "strategy": retrieval_strategy
                }
            }
    
    def _count_by_type(self, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count retrieved documents by type."""
        counts = {}
        for doc in retrieved_docs:
            doc_type = doc["element_type"]
            counts[doc_type] = counts.get(doc_type, 0) + 1
        return counts
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return self.retriever.get_stats()
    
    def export_retrieved_content(self, retrieved_content: List[Dict[str, Any]], output_dir: str) -> Dict[str, List[str]]:
        """
        Export retrieved content to files for inspection.
        
        Args:
            retrieved_content: Retrieved content from query
            output_dir: Directory to export content
            
        Returns:
            Dictionary with exported file paths by type
        """
        os.makedirs(output_dir, exist_ok=True)
        
        exported_files = {
            "images": [],
            "tables": [],
            "texts": []
        }
        
        for i, content in enumerate(retrieved_content):
            content_type = content["type"]
            
            if content_type in ["image", "graph", "page_image"] and "image_path" in content:
                # Copy image file
                import shutil
                image_path = content["image_path"]
                if os.path.exists(image_path):
                    filename = f"{content_type}_{i}_{Path(image_path).name}"
                    dest_path = os.path.join(output_dir, filename)
                    shutil.copy2(image_path, dest_path)
                    exported_files["images"].append(dest_path)
            
            elif content_type == "table" and "table_html" in content:
                # Export table as HTML
                filename = f"table_{i}.html"
                table_path = os.path.join(output_dir, filename)
                with open(table_path, "w", encoding="utf-8") as f:
                    f.write(content["table_html"])
                exported_files["tables"].append(table_path)
            
            elif content_type == "text":
                # Export text content
                filename = f"text_{i}.txt"
                text_path = os.path.join(output_dir, filename)
                with open(text_path, "w", encoding="utf-8") as f:
                    f.write(content["content"])
                exported_files["texts"].append(text_path)
        
        return exported_files

def create_multimodal_rag_system(config: Dict[str, Any] = None) -> MultimodalRAGSystem:
    """
    Factory function to create a multimodal RAG system with configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured MultimodalRAGSystem instance
    """
    if config is None:
        config = {}
    
    # Default configuration
    default_config = {
        "chroma_path": "multimodal_chroma",
        "extracted_content_path": "extracted_content",
        "text_embedding_model": "jinaai/jina-embeddings-v4",
        "multimodal_embedding_model": "jinaai/jina-embeddings-v4",
        "vlm_model": "gemma3:4b",
        "llm_model": "gemma3:4b",
        "jina_api_key": None
    }
    
    # Merge configurations
    final_config = {**default_config, **config}
    
    return MultimodalRAGSystem(**final_config)

if __name__ == "__main__":
    # Example usage
    print("Initializing Multimodal RAG System...")
    
    # Create system
    rag_system = create_multimodal_rag_system()
    
    # Get database stats
    stats = rag_system.get_database_stats()
    print(f"Database stats: {stats}")
    
    # Example: Process documents (uncomment and modify paths as needed)
    # documents = ["example.pdf", "example.docx"]
    # processing_stats = rag_system.process_documents(documents)
    # print(f"Processing completed: {processing_stats}")
    
    # Example: Query the system
    # result = rag_system.query("What are the main financial metrics?")
    # print(f"Response: {result['response']}")
    # print(f"Sources: {result['sources']}")
    
    print("Multimodal RAG System ready!") 