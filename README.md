# Multimodal RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system that handles multimodal content including text, images, graphs, and tables from PDF and DOCX documents.

## 🚀 Key Features

- **Multimodal Content Processing**: Extract and process text, images, graphs, and tables
- **Comprehensive File Format Support**: Handle 15+ file formats with specialized processing
- **Flexible Embedding Configuration**: Support for both local models and custom API endpoints
- **Dual Collection Strategy**: Separate handling of text content and page images
- **Multiple Retrieval Strategies**: Text-only, pages-only, and hybrid retrieval
- **Vision Language Model Integration**: Automatic image description generation
- **Advanced Document Processing**: PDF and DOCX support with metadata preservation
- **Web and CLI Interfaces**: Both Gradio web app and command-line interface

## 📄 Comprehensive File Format Support

### **Documents** 📄
| Format | Extension | Processing Method |
|--------|-----------|-------------------|
| **PDF** | `.pdf` | Full multimodal extraction: text chunks, images, tables, graphs, and page-level images |
| **Word** | `.doc`, `.docx` | Complete document structure extraction with embedded content |

### **Text Files** 📝
| Format | Extension | Processing Method |
|--------|-----------|-------------------|
| **Plain Text** | `.txt` | Recursive text splitting with encoding detection |
| **Rich Text** | `.rtf` | RTF-to-text conversion followed by text processing |

### **Spreadsheets** 📊
| Format | Extension | Processing Method |
|--------|-----------|-------------------|
| **CSV** | `.csv` | Convert to HTML tables with LLM-generated descriptions |
| **Excel** | `.xls`, `.xlsx` | Multi-sheet processing, each sheet as separate table element |

### **Data Files** 📋
| Format | Extension | Processing Method |
|--------|-----------|-------------------|
| **JSON** | `.json` | Smart parsing with automatic table detection for object arrays |

### **Markup & Web** 📖
| Format | Extension | Processing Method |
|--------|-----------|-------------------|
| **Markdown** | `.md`, `.markdown` | Hierarchy-aware splitting (H1→H6→paragraphs→lines) with image extraction |
| **HTML** | `.html`, `.htm` | Structure-preserving conversion with embedded image processing |

### **Images** 🖼️
| Format | Extension | Processing Method |
|--------|-----------|-------------------|
| **Standard Images** | `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.tiff`, `.webp` | VLM description generation with automatic graph/chart detection |

## 🔧 Processing Features by File Type

### **1. Text Files** (TXT, RTF)
- ✅ **Encoding Detection**: Automatic charset detection for robust text reading
- ✅ **Recursive Splitting**: Intelligent text chunking with overlap
- ✅ **Format Conversion**: RTF→Text conversion preserving content

### **2. Spreadsheets** (CSV, XLS, XLSX)
- ✅ **Multi-Sheet Support**: Each Excel sheet processed separately  
- ✅ **HTML Conversion**: Tables converted to structured HTML format
- ✅ **LLM Descriptions**: AI-generated summaries of table content and structure
- ✅ **Metadata Preservation**: Column names, data types, dimensions

### **3. JSON Files**
- ✅ **Smart Detection**: Array of objects → Table processing
- ✅ **Structure Analysis**: Nested data → Readable text format
- ✅ **Summary Generation**: Overview of JSON structure and content

### **4. Markdown Files** 
- ✅ **Hierarchy-Aware Splitting**: Respects heading structure (# → ## → ### ...)
- ✅ **Image Extraction**: Local images processed with surrounding context
- ✅ **Context Preservation**: Text before/after images maintained
- ✅ **Markdown-Specific Parsing**: Code blocks, lists, emphasis handled

### **5. HTML Files**
- ✅ **Structure-Preserving**: HTML→Text conversion maintaining hierarchy
- ✅ **Image Processing**: Embedded images with sibling context  
- ✅ **Clean Text Extraction**: Remove markup while preserving meaning

### **6. Image Files**
- ✅ **Format Support**: All major image formats (JPG, PNG, GIF, etc.)
- ✅ **VLM Processing**: Vision Language Model descriptions
- ✅ **Graph Detection**: Automatic chart/diagram recognition
- ✅ **Format Conversion**: RGBA→RGB conversion for compatibility

## ⚙️ Flexible Embedding Configuration

The system supports flexible embedding configuration that can work with:

### 🌐 Custom API Endpoints
Use your own Jina embeddings API server:
```python
config = {
    "jina_api_base_url": "http://10.144.100.204:38044",
    "jina_api_key": "your_api_key",  # Optional for custom endpoints
    "text_embedding_model": "jinaai/jina-embeddings-v4",
    "multimodal_embedding_model": "jinaai/jina-embeddings-v4"
}
```

### 🔗 Official Jina API
Use the official Jina AI API:
```python
config = {
    "jina_api_key": "your_jina_api_key",
    "text_embedding_model": "jinaai/jina-embeddings-v4",
    "multimodal_embedding_model": "jinaai/jina-embeddings-v4"
}
```

### 🏠 Local Models
Force local model usage:
```python
config = {
    "force_local_embeddings": True,
    "text_embedding_model": "jinaai/jina-embeddings-v3",
    "multimodal_embedding_model": "jinaai/jina-embeddings-v4"
}
```

## 🔧 Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `jina_api_base_url` | Custom API endpoint URL | `None` (uses official API) |
| `jina_api_key` | API key for authentication | From `JINA_API_KEY` env var |
| `force_local_embeddings` | Force local model usage | `False` |
| `text_embedding_model` | Model for text embeddings | `jinaai/jina-embeddings-v3` |
| `multimodal_embedding_model` | Model for multimodal embeddings | `jinaai/jina-embeddings-v4` |

## 🧪 Testing File Format Support

Test all supported formats:
```bash
python test_system.py
```

This will create sample files for each format and verify processing works correctly.

## 📂 Usage Examples

### Add Files to Process
```bash
# Create data directory and add your files
mkdir data
# Copy your files: PDFs, Word docs, spreadsheets, images, etc.
cp your_documents.* data/
```

### Run CLI
```bash
python cli_multimodal_rag.py
```

### Run Web Interface  
```bash
python multimodal_app.py
```

The system will automatically detect and process all supported file formats with appropriate strategies for each type. 