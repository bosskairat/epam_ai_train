import re
import os
import csv
from typing import List
from pathlib import Path
from PyPDF2 import PdfReader

# --- OCR support if needed (optional) ---
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


class PDFCleaner:
    """A module for loading, cleaning, and splitting PDF documents for RAG."""

    def __init__(self, min_block_size: int = 300, max_block_size: int = 1000):
        self.min_block_size = min_block_size
        self.max_block_size = max_block_size

    def load_pdf(self, file_path: str) -> str:
        """Extracts text from PDF (including OCR if possible)."""
        path = Path(file_path)
        text = ""

        try:
            reader = PdfReader(path)
            for page in reader.pages:
                text += page.extract_text() or ""
        except Exception:
            # fallback: If the PDF does not contain text, we apply OCR via PyMuPDF
            if fitz is None:
                raise RuntimeError("For OCR you need the PyMuPDF (fitz) package.")
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text("text")

        return text

    def clean_text(self, text: str) -> str:
        """Removes noise, special characters and normalizes format."""
        text = re.sub(r'\s+', ' ', text)                   # убираем множественные пробелы
        text = re.sub(r'-\s+', '', text)                   # переносы слов
        text = re.sub(r'Page \d+|Стр\. \d+', '', text)     # номера страниц
        text = re.sub(r'[^\x00-\x7Fа-яА-ЯёЁ\s.,;:!?()«»\'"-]', '', text)
        text = text.strip()
        return text

    def split_into_blocks(self, text: str) -> List[str]:
        """Divides the text into semantic blocks of optimal length for RAG."""
        paragraphs = re.split(r'\n{1,}|\.\s+', text)
        blocks = []
        current_block = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_block) + len(para) < self.max_block_size:
                current_block += para + ". "
            else:
                if len(current_block) >= self.min_block_size:
                    blocks.append(current_block.strip())
                    current_block = para + ". "
                else:
                    current_block += para + ". "

        if current_block:
            blocks.append(current_block.strip())

        return blocks

    def process_pdf(self, file_path: str) -> List[str]:
        """Full cycle: loading → cleaning → partitioning."""
        raw_text = self.load_pdf(file_path)
        clean_text = self.clean_text(raw_text)
        blocks = self.split_into_blocks(clean_text)
        print(f"✅ Processed: {len(blocks)} blocks extracted from {file_path}")
        return blocks


# --- Processing ---
if __name__ == "__main__":
    
    path = "./data/pdf"
    output_csv = "./data/article_chunks.csv"           # куда сохранить

    # CSV template
    header = ["file", "chunk_id", "content"]

    with open(output_csv, "w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(header)

        for root, dirs, files in os.walk(path):
            for f in files:
                if f.lower().endswith(".pdf"):

                    pdf_path = os.path.join(root, f)
                    print(f"\nProcessing the file: {pdf_path}")

                    cleaner = PDFCleaner(min_block_size=200, max_block_size=1000)
                    chunks = cleaner.process_pdf(pdf_path)

                    for i, c in enumerate(chunks, start=1):
                        # write to CSV
                        writer.writerow([os.path.basename(pdf_path), i, c])

                    print(f" -> saved {len(chunks)} chunks")