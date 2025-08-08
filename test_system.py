#!/usr/bin/env python3
"""
Test suite for the multimodal RAG system with comprehensive file format support.
"""

import os
import sys
import logging
import tempfile
import json
from pathlib import Path
from PIL import Image, ImageDraw
import pandas as pd

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
    
def test_file_format_support():
    """Test all supported file formats"""
    print("🔬 Testing Multimodal RAG System File Format Support")
    print("=" * 60)
    
    # Test imports
    try:
        from multimodal_document_processor import MultimodalDocumentProcessor
        print("✓ Successfully imported MultimodalDocumentProcessor")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    
    # Create test files
    test_dir = Path("test_files")
    test_dir.mkdir(exist_ok=True)
    
    print(f"\n📁 Creating test files in {test_dir}...")
    
    # Create test files for each format
    test_files = {}
    
    # 1. TXT file
    txt_file = test_dir / "test.txt"
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("This is a test text file.\n\nIt contains multiple paragraphs.\n\nAnd demonstrates text processing capabilities.")
    test_files['txt'] = txt_file
    print("✓ Created test.txt")
    
    # 2. RTF file (simple RTF format)
    rtf_file = test_dir / "test.rtf"
    with open(rtf_file, 'w', encoding='utf-8') as f:
        f.write(r"""{\rtf1\ansi\deff0 {\fonttbl {\f0 Times New Roman;}}
\f0\fs24 This is a test RTF document.
\par
\b Bold text \b0 and \i italic text \i0.
\par
Multiple paragraphs for testing.
}""")
    test_files['rtf'] = rtf_file
    print("✓ Created test.rtf")
    
    # 3. CSV file
    csv_file = test_dir / "test.csv"
    df = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'London', 'Tokyo']
    })
    df.to_csv(csv_file, index=False)
    test_files['csv'] = csv_file
    print("✓ Created test.csv")
    
    # 4. Excel file
    xlsx_file = test_dir / "test.xlsx"
    with pd.ExcelWriter(xlsx_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='People', index=False)
        df2 = pd.DataFrame({'Product': ['A', 'B', 'C'], 'Price': [10, 20, 30]})
        df2.to_excel(writer, sheet_name='Products', index=False)
    test_files['xlsx'] = xlsx_file
    print("✓ Created test.xlsx")
    
    # 5. JSON file
    json_file = test_dir / "test.json"
    json_data = {
        "users": [
            {"id": 1, "name": "Alice", "email": "alice@example.com"},
            {"id": 2, "name": "Bob", "email": "bob@example.com"}
        ],
        "metadata": {
            "version": "1.0",
            "created": "2024-01-01"
        }
    }
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)
    test_files['json'] = json_file
    print("✓ Created test.json")
    
    # 6. Markdown file
    md_file = test_dir / "test.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("""# Test Markdown Document

## Introduction
This is a test markdown file with **bold** and *italic* text.

### Features
- List item 1
- List item 2
- List item 3

#### Code Example
```python
print("Hello, World!")
```

## Conclusion
This demonstrates markdown processing capabilities.
""")
    test_files['md'] = md_file
    print("✓ Created test.md")
    
    # 7. HTML file
    html_file = test_dir / "test.html"
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Test HTML Document</title>
</head>
<body>
    <h1>Test HTML Document</h1>
    <h2>Introduction</h2>
    <p>This is a test HTML file with <strong>bold</strong> and <em>italic</em> text.</p>
    
    <h3>Features</h3>
    <ul>
        <li>HTML parsing</li>
        <li>Text extraction</li>
        <li>Structure preservation</li>
    </ul>
    
    <h2>Conclusion</h2>
    <p>This demonstrates HTML processing capabilities.</p>
</body>
</html>""")
    test_files['html'] = html_file
    print("✓ Created test.html")
    
    # 8. Image file
    png_file = test_dir / "test.png"
    img = Image.new('RGB', (200, 100), color='lightblue')
    draw = ImageDraw.Draw(img)
    draw.text((10, 40), "Test Image", fill='black')
    img.save(png_file)
    test_files['png'] = png_file
    print("✓ Created test.png")
    
    print(f"\n🔧 Testing document processor...")
    
    # Initialize processor
    processor = MultimodalDocumentProcessor()
    
    # Test each file format
    results = {}
    for format_name, file_path in test_files.items():
        print(f"\n📄 Testing {format_name.upper()} format: {file_path.name}")
        try:
            elements = processor.process_document(str(file_path), "test_output")
            results[format_name] = {
                'success': True,
                'elements_count': len(elements),
                'element_types': [e.element_type for e in elements]
            }
            print(f"   ✓ Processed successfully: {len(elements)} elements")
            print(f"   ✓ Element types: {set(e.element_type for e in elements)}")
            
            # Convert to documents
            documents = processor.elements_to_documents(elements)
            print(f"   ✓ Converted to {len(documents)} documents")
            
        except Exception as e:
            results[format_name] = {
                'success': False,
                'error': str(e)
            }
            print(f"   ✗ Failed: {e}")
    
    # Summary
    print(f"\n📊 Test Results Summary:")
    print("=" * 60)
    
    successful = sum(1 for r in results.values() if r['success'])
    total = len(results)
    
    for format_name, result in results.items():
        status = "✓ PASS" if result['success'] else "✗ FAIL"
        print(f"{format_name.upper():8} {status}")
        if result['success']:
            print(f"         Elements: {result['elements_count']}, Types: {set(result['element_types'])}")
        else:
            print(f"         Error: {result['error']}")
    
    print(f"\nOverall: {successful}/{total} formats working correctly")
    
    if successful == total:
        print("\n🎉 All file formats are working correctly!")
    else:
        print(f"\n⚠️  {total - successful} formats need attention.")
    
    return successful == total

if __name__ == "__main__":
    success = test_file_format_support()
    exit(0 if success else 1) 