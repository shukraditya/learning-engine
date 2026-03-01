#!/usr/bin/env python3
"""Test script: Extract first 10 pages and convert to Markdown."""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from core.tree_engine import DocumentProcessor


def main():
    pdf_path = Path("../Database Internals.pdf")
    test_pdf = Path("/tmp/test_10_pages.pdf")
    output_md = Path("/tmp/test_output.md")

    print(f"📄 Extracting first 10 pages from: {pdf_path}")

    # Extract first 10 pages using pypdf
    try:
        from pypdf import PdfReader, PdfWriter

        reader = PdfReader(str(pdf_path))
        writer = PdfWriter()

        num_pages = min(10, len(reader.pages))
        for i in range(num_pages):
            writer.add_page(reader.pages[i])

        with open(test_pdf, "wb") as f:
            writer.write(f)

        print(f"✓ Created test PDF: {test_pdf} ({num_pages} pages)")

    except ImportError:
        print("⚠️  pypdf not installed, trying with PyMuPDF...")
        import fitz  # PyMuPDF

        doc = fitz.open(str(pdf_path))
        new_doc = fitz.open()

        num_pages = min(10, len(doc))
        for i in range(num_pages):
            new_doc.insert_pdf(doc, from_page=i, to_page=i)

        new_doc.save(str(test_pdf))
        new_doc.close()
        doc.close()

        print(f"✓ Created test PDF: {test_pdf} ({num_pages} pages)")

    # Convert to Markdown using Marker
    print(f"\n🔄 Converting to Markdown with Marker...")
    print("   (First run will download models ~5GB)")
    print()

    processor = DocumentProcessor(verbose=True)

    try:
        pages_meta, md_content = processor.pdf_to_markdown(test_pdf, output_md)

        print(f"\n✓ Conversion complete!")
        print(f"  Pages: {pages_meta.get('page_count', '?')}")
        print(f"  Markdown: {len(md_content)} chars")
        print(f"  Output: {output_md}")
        print()
        print("📋 First 500 characters:")
        print("-" * 50)
        print(md_content[:500])
        print("-" * 50)

    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()