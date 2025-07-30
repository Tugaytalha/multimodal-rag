import os
import io
import json
import base64
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import hashlib
import logging

# Document processing
import fitz  # PyMuPDF for PDF processing
from docx import Document as DocxDocument
from docx.document import Document as DocxDocumentType
from PIL import Image
import pandas as pd

# Langchain components
from langchain.schema.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# HTML processing for tables
from bs4 import BeautifulSoup
import html2text

# Vision and LLM processing
from transformers import AutoProcessor, AutoModelForCausalLM, pipeline
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExtractedElement:
    """Container for extracted document elements."""
    element_type: str  # 'text', 'image', 'table', 'graph'
    content: Any  # Text content, image bytes, HTML table, etc.
    metadata: Dict[str, Any]
    page_number: int
    element_id: str
    bbox: Optional[Tuple[float, float, float, float]] = None  # Bounding box coordinates

class MultimodalDocumentProcessor:
    """
    Multimodal document processor that extracts and processes different types of content
    from documents (PDF, DOCX) according to specific processing strategies.
    """
    
    def __init__(self, 
                 vlm_model_name: str = "microsoft/git-base-coco",
                 chunk_size: int = 800,
                 chunk_overlap: int = 80):
        """
        Initialize the multimodal document processor.
        
        Args:
            vlm_model_name: Vision Language Model for image description
            chunk_size: Text chunk size for splitting
            chunk_overlap: Overlap between text chunks
        """
        self.vlm_model_name = vlm_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        
        # Initialize VLM for image description
        self._init_vlm()
        
        # Initialize HTML converter for tables
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = True
        self.html_converter.ignore_images = True
        
    def _init_vlm(self):
        """Initialize Vision Language Model (VLM)."""
        if not self.vlm_model_name:
            logger.warning("No VLM model specified")
            return

        try:
            logger.info(f"Loading VLM model: {self.vlm_model_name}")
            
            # Check if this is an Ollama model (contains colon)
            if ":" in self.vlm_model_name or self.vlm_model_name.startswith("gemma3") or self.vlm_model_name.startswith("llava") or self.vlm_model_name.startswith("llama3.2-vision"):
                # Use Ollama for models like gemma3:4b, llava:7b, etc.
                logger.info(f"Using Ollama for VLM: {self.vlm_model_name}")
                
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
                            logger.error("Could not import Ollama. Please install langchain-community or langchain-ollama")
                            return

                self.vlm_model = Ollama(model=self.vlm_model_name)
                self.vlm_processor = None  # Not needed for Ollama
                self.use_ollama_vlm = True
                logger.info(f"Ollama VLM model loaded: {self.vlm_model_name}")
                
            else:
                # Use HuggingFace transformers for models like microsoft/git-base-coco
                logger.info(f"Using HuggingFace for VLM: {self.vlm_model_name}")
                from transformers import AutoProcessor, AutoModelForCausalLM
                
                self.vlm_processor = AutoProcessor.from_pretrained(self.vlm_model_name)
                self.vlm_model = AutoModelForCausalLM.from_pretrained(self.vlm_model_name)
                self.vlm_model.to(self.device)
                self.use_ollama_vlm = False
                logger.info(f"HuggingFace VLM model loaded: {self.vlm_model_name}")
                
        except Exception as e:
            logger.error(f"Failed to load VLM model: {e}")
            self.vlm_model = None
            self.vlm_processor = None
            self.use_ollama_vlm = False
    
    def _generate_element_id(self, content: Any, page_num: int, element_type: str) -> str:
        """Generate unique ID for extracted elements."""
        content_hash = hashlib.md5(str(content).encode()).hexdigest()[:8]
        return f"{element_type}_{page_num}_{content_hash}"
    
    def _describe_image_with_vlm(self, image: Image.Image, surrounding_text: str = "") -> str:
        """Generate description for image using VLM."""
        if not self.vlm_model:
            return "Image description unavailable (VLM not loaded)"

        try:
            if hasattr(self, 'use_ollama_vlm') and self.use_ollama_vlm:
                # Use Ollama for multimodal description
                return self._describe_image_with_ollama(image, surrounding_text)
            else:
                # Use HuggingFace transformers
                return self._describe_image_with_huggingface(image, surrounding_text)
                
        except Exception as e:
            logger.error(f"Error generating image description: {e}")
            return f"Error generating image description: {str(e)}"

    def _describe_image_with_ollama(self, image: Image.Image, surrounding_text: str = "") -> str:
        """Generate description using Ollama VLM."""
        try:
            import tempfile
            import os
            
            # Save image temporarily for Ollama
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                image.save(tmp_file.name, format='PNG')
                temp_image_path = tmp_file.name
            
            try:
                # Create prompt for Ollama multimodal models
                context_text = f"Context: {surrounding_text[:200]}" if surrounding_text else ""
                prompt = f"""Please describe this image in detail. Focus on the main subjects, activities, text content, charts, graphs, or any other important visual elements.

{context_text}

What do you see in this image?"""

                # For Ollama multimodal, we need to check if it supports direct API calls
                # If using ollama Python library directly:
                try:
                    import ollama
                    response = ollama.chat(
                        model=self.vlm_model_name,
                        messages=[{
                            'role': 'user',
                            'content': prompt,
                            'images': [temp_image_path]
                        }]
                    )
                    description = response['message']['content']
                except ImportError:
                    # Fallback to langchain if ollama library not available
                    # Note: This might not work for multimodal, depends on langchain implementation
                    logger.warning("ollama library not found, falling back to langchain (multimodal support limited)")
                    description = self.vlm_model.invoke(prompt)
                
                return description.strip() if description else "Could not generate image description"
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_image_path):
                    os.unlink(temp_image_path)
            
        except Exception as e:
            logger.error(f"Error with Ollama VLM: {e}")
            return f"Image description unavailable (Ollama error: {str(e)})"

    def _describe_image_with_huggingface(self, image: Image.Image, surrounding_text: str = "") -> str:
        """Generate description using HuggingFace transformers."""
        try:
            prompt_text = f"Describe this image in detail. Context: {surrounding_text[:200]}" if surrounding_text else "Describe this image in detail."
            
            # Different models have different input formats
            if "git" in self.vlm_model_name.lower():
                # For GIT models
                inputs = self.vlm_processor(images=image, text=prompt_text, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    generated_ids = self.vlm_model.generate(**inputs, max_length=150, num_beams=4, early_stopping=True)
                description = self.vlm_processor.decode(generated_ids[0], skip_special_tokens=True)
            elif "blip" in self.vlm_model_name.lower():
                # For BLIP models
                inputs = self.vlm_processor(image, prompt_text, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    generated_ids = self.vlm_model.generate(**inputs, max_length=150)
                description = self.vlm_processor.decode(generated_ids[0], skip_special_tokens=True)
            else:
                # Generic approach
                inputs = self.vlm_processor(images=image, text=prompt_text, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    generated_ids = self.vlm_model.generate(**inputs, max_length=150, num_beams=4, early_stopping=True)
                description = self.vlm_processor.decode(generated_ids[0], skip_special_tokens=True)
            
            # Clean up the description
            if prompt_text in description:
                description = description.replace(prompt_text, "").strip()
            
            return description if description else "Could not generate image description"
            
        except Exception as e:
            logger.error(f"Error with HuggingFace VLM: {e}")
            return f"Image description unavailable (HF error: {str(e)})"
    
    def _extract_table_description(self, table_html: str, surrounding_text: str = "") -> str:
        """
        Generate description for a table using LLM.
        
        Args:
            table_html: HTML representation of the table
            surrounding_text: Text context around the table
            
        Returns:
            Description of the table
        """
        try:
            # Convert HTML to readable text
            table_text = self.html_converter.handle(table_html)
            
            # Create prompt for table description
            prompt = f"""
            Analyze the following table and provide a comprehensive description including:
            1. The main purpose and content of the table
            2. Key columns and their meanings
            3. Important data patterns or trends
            4. Any notable values or relationships
            
            Context: {surrounding_text[:200]}
            
            Table:
            {table_text[:1000]}
            
            Description:
            """
            
            # Use a simple text generation approach (can be replaced with more sophisticated LLM)
            # For now, return a structured description
            return f"Table containing {table_text.count('|')} columns with data about {surrounding_text[:100] if surrounding_text else 'various metrics'}. Full table content: {table_text[:500]}"
            
        except Exception as e:
            logger.error(f"Error generating table description: {e}")
            return f"Table description error: {str(e)}"
    
    def _save_image(self, image: Image.Image, save_dir: str, filename: str) -> str:
        """Save image to disk and return path."""
        os.makedirs(save_dir, exist_ok=True)
        image_path = os.path.join(save_dir, filename)
        image.save(image_path)
        return image_path
    
    def _save_page_as_image(self, page, save_dir: str, filename: str) -> str:
        """Save PDF page as image."""
        os.makedirs(save_dir, exist_ok=True)
        # Render page as image
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
        img_data = pix.tobytes("png")
        
        # Save image
        image_path = os.path.join(save_dir, filename)
        with open(image_path, "wb") as f:
            f.write(img_data)
        
        return image_path
    
    def process_pdf(self, pdf_path: str, output_dir: str = "extracted_content") -> List[ExtractedElement]:
        """
        Process PDF file and extract different types of content.
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save extracted images
            
        Returns:
            List of extracted elements
        """
        elements = []
        doc_name = Path(pdf_path).stem
        image_dir = os.path.join(output_dir, "images", doc_name)
        page_image_dir = os.path.join(output_dir, "pages", doc_name)
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Save entire page as image
                page_image_filename = f"page_{page_num + 1}.png"
                page_image_path = self._save_page_as_image(page, page_image_dir, page_image_filename)
                
                # Create page image element
                page_element = ExtractedElement(
                    element_type="page_image",
                    content=page_image_path,
                    metadata={
                        "source": pdf_path,
                        "page": page_num + 1,
                        "document_name": doc_name,
                        "image_path": page_image_path,
                        "content_type": "page_image"
                    },
                    page_number=page_num + 1,
                    element_id=self._generate_element_id(page_image_path, page_num + 1, "page_image")
                )
                elements.append(page_element)
                
                # Extract text
                text = page.get_text()
                if text.strip():
                    text_element = ExtractedElement(
                        element_type="text",
                        content=text,
                        metadata={
                            "source": pdf_path,
                            "page": page_num + 1,
                            "document_name": doc_name
                        },
                        page_number=page_num + 1,
                        element_id=self._generate_element_id(text, page_num + 1, "text")
                    )
                    elements.append(text_element)
                
                # Extract images
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        if pix.n - pix.alpha < 4:  # Ensure it's not CMYK
                            img_data = pix.tobytes("png")
                            image = Image.open(io.BytesIO(img_data))
                            
                            # Save image
                            img_filename = f"img_{page_num + 1}_{img_index}.png"
                            img_path = self._save_image(image, image_dir, img_filename)
                            
                            # Generate description
                            surrounding_text = text[:500] if text else ""
                            description = self._describe_image_with_vlm(image, surrounding_text)
                            
                            # Determine if it's a graph or regular image
                            is_graph = any(keyword in description.lower() 
                                         for keyword in ['chart', 'graph', 'plot', 'diagram', 'visualization'])
                            element_type = "graph" if is_graph else "image"
                            
                            img_element = ExtractedElement(
                                element_type=element_type,
                                content=img_path,
                                metadata={
                                    "source": pdf_path,
                                    "page": page_num + 1,
                                    "document_name": doc_name,
                                    "image_path": img_path,
                                    "description": description,
                                    "surrounding_text": surrounding_text,
                                    "content_type": element_type
                                },
                                page_number=page_num + 1,
                                element_id=self._generate_element_id(img_path, page_num + 1, element_type)
                            )
                            elements.append(img_element)
                            
                        pix = None
                    except Exception as e:
                        logger.error(f"Error processing image {img_index} on page {page_num + 1}: {e}")
                
                # Extract tables
                tables = page.find_tables()
                for table_index, table in enumerate(tables):
                    try:
                        # Extract table as pandas DataFrame
                        df = table.to_pandas()
                        
                        # Convert to HTML
                        table_html = df.to_html(index=False, escape=False)
                        
                        # Generate description
                        surrounding_text = text[:500] if text else ""
                        description = self._extract_table_description(table_html, surrounding_text)
                        
                        table_element = ExtractedElement(
                            element_type="table",
                            content=table_html,
                            metadata={
                                "source": pdf_path,
                                "page": page_num + 1,
                                "document_name": doc_name,
                                "description": description,
                                "surrounding_text": surrounding_text,
                                "table_shape": df.shape,
                                "content_type": "table"
                            },
                            page_number=page_num + 1,
                            element_id=self._generate_element_id(table_html, page_num + 1, "table"),
                            bbox=table.bbox
                        )
                        elements.append(table_element)
                        
                    except Exception as e:
                        logger.error(f"Error processing table {table_index} on page {page_num + 1}: {e}")
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            raise
        
        return elements
    
    def process_docx(self, docx_path: str, output_dir: str = "extracted_content") -> List[ExtractedElement]:
        """
        Process DOCX file and extract different types of content.
        
        Args:
            docx_path: Path to DOCX file
            output_dir: Directory to save extracted images
            
        Returns:
            List of extracted elements
        """
        elements = []
        doc_name = Path(docx_path).stem
        image_dir = os.path.join(output_dir, "images", doc_name)
        
        try:
            doc = DocxDocument(docx_path)
            
            # Extract text content
            full_text = ""
            for paragraph in doc.paragraphs:
                full_text += paragraph.text + "\n"
            
            if full_text.strip():
                text_element = ExtractedElement(
                    element_type="text",
                    content=full_text,
                    metadata={
                        "source": docx_path,
                        "document_name": doc_name,
                        "page": 1  # DOCX doesn't have clear page boundaries
                    },
                    page_number=1,
                    element_id=self._generate_element_id(full_text, 1, "text")
                )
                elements.append(text_element)
            
            # Extract images from DOCX
            from docx.document import Document as DocxDoc
            import zipfile
            
            # Extract images from the DOCX file structure
            with zipfile.ZipFile(docx_path, 'r') as docx_zip:
                image_files = [f for f in docx_zip.namelist() if f.startswith('word/media/')]
                
                for img_index, img_file in enumerate(image_files):
                    try:
                        img_data = docx_zip.read(img_file)
                        image = Image.open(io.BytesIO(img_data))
                        
                        # Save image
                        img_filename = f"img_{img_index}.png"
                        img_path = self._save_image(image, image_dir, img_filename)
                        
                        # Generate description
                        description = self._describe_image_with_vlm(image, full_text[:500])
                        
                        # Determine if it's a graph or regular image
                        is_graph = any(keyword in description.lower() 
                                     for keyword in ['chart', 'graph', 'plot', 'diagram', 'visualization'])
                        element_type = "graph" if is_graph else "image"
                        
                        img_element = ExtractedElement(
                            element_type=element_type,
                            content=img_path,
                            metadata={
                                "source": docx_path,
                                "document_name": doc_name,
                                "image_path": img_path,
                                "description": description,
                                "surrounding_text": full_text[:500],
                                "content_type": element_type,
                                "page": 1
                            },
                            page_number=1,
                            element_id=self._generate_element_id(img_path, 1, element_type)
                        )
                        elements.append(img_element)
                        
                    except Exception as e:
                        logger.error(f"Error processing image {img_index} in DOCX: {e}")
            
            # Extract tables from DOCX
            for table_index, table in enumerate(doc.tables):
                try:
                    # Convert table to HTML
                    table_data = []
                    for row in table.rows:
                        row_data = [cell.text for cell in row.cells]
                        table_data.append(row_data)
                    
                    df = pd.DataFrame(table_data[1:], columns=table_data[0] if table_data else [])
                    table_html = df.to_html(index=False, escape=False)
                    
                    # Generate description
                    description = self._extract_table_description(table_html, full_text[:500])
                    
                    table_element = ExtractedElement(
                        element_type="table",
                        content=table_html,
                        metadata={
                            "source": docx_path,
                            "document_name": doc_name,
                            "description": description,
                            "surrounding_text": full_text[:500],
                            "table_shape": df.shape,
                            "content_type": "table",
                            "page": 1
                        },
                        page_number=1,
                        element_id=self._generate_element_id(table_html, 1, "table")
                    )
                    elements.append(table_element)
                    
                except Exception as e:
                    logger.error(f"Error processing table {table_index} in DOCX: {e}")
        
        except Exception as e:
            logger.error(f"Error processing DOCX {docx_path}: {e}")
            raise
        
        return elements
    
    def process_document(self, file_path: str, output_dir: str = "extracted_content") -> List[ExtractedElement]:
        """
        Process a document file and extract all content types.
        
        Args:
            file_path: Path to document file
            output_dir: Directory to save extracted content
            
        Returns:
            List of extracted elements
        """
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.pdf':
            return self.process_pdf(file_path, output_dir)
        elif file_extension in ['.docx', '.doc']:
            return self.process_docx(file_path, output_dir)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def elements_to_documents(self, elements: List[ExtractedElement]) -> List[Document]:
        """
        Convert extracted elements to Langchain Documents based on their type.
        
        Args:
            elements: List of extracted elements
            
        Returns:
            List of Langchain Documents ready for embedding
        """
        documents = []
        
        for element in elements:
            if element.element_type == "text":
                # Split text into chunks
                text_docs = self.text_splitter.create_documents(
                    [element.content],
                    metadatas=[element.metadata]
                )
                
                # Add unique IDs to text chunks
                for i, doc in enumerate(text_docs):
                    doc.metadata["id"] = f"{element.element_id}_chunk_{i}"
                    doc.metadata["element_type"] = "text"
                    doc.metadata["original_element_id"] = element.element_id
                
                documents.extend(text_docs)
                
            elif element.element_type in ["image", "graph"]:
                # For images and graphs, embed the description but keep image path for retrieval
                doc = Document(
                    page_content=element.metadata.get("description", ""),
                    metadata={
                        **element.metadata,
                        "id": element.element_id,
                        "element_type": element.element_type,
                        "image_path": element.content,
                        "retrieve_content": element.content  # Store image path for retrieval
                    }
                )
                documents.append(doc)
                
            elif element.element_type == "table":
                # For tables, embed the description but keep HTML structure
                doc = Document(
                    page_content=element.metadata.get("description", ""),
                    metadata={
                        **element.metadata,
                        "id": element.element_id,
                        "element_type": "table",
                        "table_html": element.content,
                        "retrieve_content": element.content  # Store HTML for retrieval
                    }
                )
                documents.append(doc)
                
            elif element.element_type == "page_image":
                # Page images will be handled separately with multimodal embeddings
                doc = Document(
                    page_content=f"Page {element.page_number} image",
                    metadata={
                        **element.metadata,
                        "id": element.element_id,
                        "element_type": "page_image",
                        "image_path": element.content,
                        "retrieve_content": element.content
                    }
                )
                documents.append(doc)
        
        return documents

if __name__ == "__main__":
    # Example usage
    processor = MultimodalDocumentProcessor()
    
    # Process a document
    file_path = "example.pdf"  # Replace with actual file path
    elements = processor.process_document(file_path)
    
    print(f"Extracted {len(elements)} elements:")
    for element in elements:
        print(f"- {element.element_type}: {element.element_id}")
    
    # Convert to documents
    documents = processor.elements_to_documents(elements)
    print(f"Created {len(documents)} documents for embedding") 