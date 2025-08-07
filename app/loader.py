import os
from PyPDF2 import PdfReader
from docx import Document as DocxReader

def load_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        reader = PdfReader(file_path)
        return [page.extract_text() for page in reader.pages]

    elif ext == ".docx":
        doc = DocxReader(file_path)
        return [para.text for para in doc.paragraphs if para.text.strip()]

    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().split("\n\n")  # basic chunking

    elif ext in [".eml", ".email"]:
        with open(file_path, "r", encoding="utf-8") as f:
            return [f.read()]  # treat entire email as one chunk

    else:
        raise ValueError(f"Unsupported file format: {ext}")
