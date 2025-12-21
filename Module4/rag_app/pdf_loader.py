import re
import os
import csv
from typing import List
from pathlib import Path
from pypdf import PdfReader


class PDFCleaner:
    """A module for loading, cleaning, and splitting PDF documents for RAG."""

    def __init__(self, min_block_size: int = 300, max_block_size: int = 1000, overlap_sentences: int = 2):
        self.min_block_size = min_block_size
        self.max_block_size = max_block_size
        self.overlap_sentences = max(0, int(overlap_sentences))

    def load_pdf(self, file_path: str) -> str:
        """Extracts text from PDF (including OCR if possible)."""
        path = Path(file_path)
        text = ""

        try:
            reader = PdfReader(path)
            for page in reader.pages:
                text += page.extract_text() or ""
        except Exception:
            print(f"⚠️  Warning: Could not read PDF file {file_path}.")            

        return text

    def clean_text(self, text: str) -> str:
        """Removes noise, special characters and normalizes format."""
        text = re.sub(r'\s+', ' ', text)                   # убираем множественные пробелы
        text = re.sub(r'-\s+', '', text)                   # переносы слов
        text = re.sub(r'Page \d+|Стр\. \d+', '', text)     # номера страниц
        text = re.sub(r'[^\x00-\x7Fа-яА-ЯёЁ\s.,;:!?()«»\'"-]', '', text)
        text = text.strip()
        return text

    # Baseline splitting method
    # def split_into_blocks(self, text: str) -> List[str]:
    #     """Divides the text into semantic blocks of optimal length for RAG."""
    #     paragraphs = re.split(r'\n{1,}|\.\s+', text)
    #     blocks = []
    #     current_block = ""

    #     for para in paragraphs:
    #         para = para.strip()
    #         if not para:
    #             continue

    #         if len(current_block) + len(para) < self.max_block_size:
    #             current_block += para + ". "
    #         else:
    #             if len(current_block) >= self.min_block_size:
    #                 blocks.append(current_block.strip())
    #                 current_block = para + ". "
    #             else:
    #                 current_block += para + ". "

    #     if current_block:
    #         blocks.append(current_block.strip())

    #     return blocks
    
    
    # Improved splitting method with sentence-based chunks and overlap
    def split_into_blocks(self, text: str) -> List[str]:
        """Create sentence-based chunks with overlap for better retrieval.

        Strategy:
        - Split text into sentences.
        - Build chunks by aggregating sentences until reaching `max_block_size`.
        - Ensure each chunk is at least `min_block_size` when possible.
        - Advance the window by (chunk_sentence_count - overlap_sentences) to create overlaps.
        """
        # Split into sentences (keeps punctuation at sentence end)
        sentences = [s.strip() for s in re.split(r'(?<=[\.\!\?…])\s+', text) if s.strip()]
        blocks: List[str] = []
        i = 0
        n = len(sentences)

        while i < n:
            chunk = ""
            j = i
            # Add sentences while under max size
            while j < n and len(chunk) + len(sentences[j]) + 1 <= self.max_block_size:
                chunk += sentences[j] + " "
                j += 1

            # If no sentence fit because a single sentence is longer than max, truncate it
            if j == i:
                long_sentence = sentences[j]
                chunk = long_sentence[: self.max_block_size].strip()
                j = i + 1

            # Try to grow chunk to reach min_block_size if possible
            if len(chunk) < self.min_block_size and j < n:
                while j < n and len(chunk) < self.min_block_size:
                    chunk += sentences[j] + " "
                    j += 1

            blocks.append(chunk.strip())

            # Advance start index to create overlap; ensure progress by at least 1 sentence
            sentences_in_chunk = max(1, j - i)
            step = max(1, sentences_in_chunk - self.overlap_sentences)
            i = i + step

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