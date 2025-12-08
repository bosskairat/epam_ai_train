# Question-Answering System Based on Pre-Saved PDF Documents

## Project Overview
This project implements a **Question-Answering (QA) System** that allows users to query documents stored in a local directory. Using **RAG (Retrieval-Augmented Generation)**, the system retrieves relevant text blocks from PDFs and generates answers using an LLM.

---

## Main Idea
- Users can query a collection of PDF documents stored locally.  
- The system extracts text from PDFs, splits it into semantic blocks, generates embeddings, and stores them in a vector database.  
- When a query is made, the system retrieves the most relevant blocks and passes them to an LLM to generate an answer.  
- A simple interface allows users to type queries and view AI-generated answers.

---

## Concepts
- **RAG (Retrieval-Augmented Generation):** Combines vector search with LLMs to answer questions based on document content.  
- **PDF Processing:** Extracts text, cleans, and splits it into blocks suitable for semantic search.  
- **Embeddings:** Each block is converted into a vector representing its semantic meaning.  
- **Vector Database:** Stores vectors for efficient similarity search and retrieval.  
- **LLM:** Generates natural language answers from retrieved context.  
- **UI:** Minimal WebApp for entering queries and displaying results.

---

## Design Details
1. **Documents stored in a directory** are processed using a PDF loader script.  
2. **Text is cleaned and split** into semantic blocks (200–1000 characters per block).  
3. **CSV storage:** Blocks are saved in `article_chunks.csv` with columns: `file`, `chunk_id`, `content`.  
4. **Embeddings generation:** A local HuggingFace embedding model (`LocalHuggingFaceEmbeddings`) converts each block to a vector.  
5. **Vector Database (Weaviate):**  
   - Run as a Docker container (`semitechnologies/weaviate:1.33.7`)  
   - Anonymous access enabled, persistence enabled, no default vectorizer  
   - Supports vector similarity search using HNSW and cosine distance  
6. **Weaviate Collection Setup:**  
   - Check if collection exists; delete if present  
   - Create a new collection with properties: `file`, `chunk_id`, `content`  
   - Configure self-provided embeddings (vectors generated locally)  
7. **Batch Data Ingestion:**  
   - Each document block with its vector is ingested into the collection  
   - UUID generated deterministically based on `file` + `chunk_id`  
8. **RAG Query Pipeline:**  
   - User query is expanded using `ChatPromptTemplate` to improve search relevance  
   - Query embedding generated with the same embedding model  
   - Top matching blocks retrieved from Weaviate  
   - Retrieved blocks passed to local LLM (`LocalHuggingFaceChatModel`)  
   - LLM generates a concise, bullet-point answer based only on retrieved context  

---

## Dataset Concept
- **Content:** Pre-saved PDF files in `./data/pdf`.  
- **Annotations:** Each block linked to its source file and block ID.  
- **Purpose:** Provide representative content for testing retrieval and QA.  
- **Format:** CSV file (`article_chunks.csv`) storing `file`, `chunk_id`, `content`.  

---

## System Technical Details
- **Vector Database:** Weaviate (Docker container)  
- **Embeddings Model:** Local HuggingFace model (`google/embeddinggemma-300m`)  
- **LLM Model:** Local HuggingFace chat model (`google/gemma-3-1b-it`)  
- **PDF Loader:** Python script using `PyPDF2` (with optional OCR via `PyMuPDF`)  
- **Data Ingestion Script:**  
  - Loads document chunks from CSV  
  - Generates embeddings  
  - Connects to Weaviate  
  - Creates collection and batch inserts data  
- **RAG Pipeline Script:**  
  - Expands user query for semantic search  
  - Retrieves top relevant documents from Weaviate  
  - Generates final answer using LLM  
- **UI:** CLI or Streamlit for query input and answer display  
- **Environment:** Python 3.10+, Docker (for database)

---

## Requirements
- Python 3.10+  
- Docker (for Weaviate)  
- Libraries: `fastapi`, `PyPDF2`, `fitz` (optional), `langchain`, `weaviate-client`, `huggingface-hub`, `transformers`, `sentence-transformers`  
- GPU recommended for faster embeddings and LLM inference (optional)   

---

## Limitations
- Only PDF documents are supported currently.  
- Very large documents may exceed RAG context window limits.  
- Accuracy depends on quality of embeddings and LLM.  
- Current system optimized for small to medium datasets; large-scale usage may require optimization.  

---

## Future Improvements
- Add support for DOCX, TXT, or web-sourced documents.  
- Improve PDF chunking and semantic search accuracy.  
- Integrate federated learning for collaborative knowledge bases.  
- Scale system for larger datasets and multiple users.
- Add cloud Embedings and LLM models


## Test questions
- What methods or techniques are described in the articles?
- Explain the meaning of [specific term] mentioned in the articles.
- What numbers, statistics, or results are reported in the articles?

## Project Video
A demonstration of the workflow—PDF processing, embeddings generation, Weaviate ingestion, RAG query expansion, and LLM answer generation—is available here:  
[https://youtu.be/Fulcq0f8xeU](#) 
[https://github.com/bosskairat/epam_ai_train/tree/main/Module3/rag_app](#)