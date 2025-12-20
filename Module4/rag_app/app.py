from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from contextlib import asynccontextmanager
from rag_system import RAG
import shutil
import csv
from pathlib import Path
from pdf_loader import PDFCleaner
import uvicorn

templates = Jinja2Templates(directory="templates")

# Request model
class QueryRequest(BaseModel):
    question: str

# Lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: создаём RAG
    app.state.rag_system = RAG()
    yield
    # Shutdown: закрываем Weaviate и очищаем память
    app.state.rag_system.close_weaviate_client()

app = FastAPI(title="RAG API with Web UI", lifespan=lifespan)

# API endpoint
@app.post("/ask")
async def ask_question(query_request: QueryRequest):
    rag_system: RAG = app.state.rag_system
    try:
        expanded_query, context, answer = rag_system.answer_the_question(query_request.question)
        return {"question": query_request.question, 
                "expanded_query": expanded_query, 
                "context":context, 
                "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# New: Upload endpoint for multiple PDF files
@app.post("/upload")
async def upload_pdf(files: list[UploadFile] = File(...)):
    base_dir = Path(__file__).resolve().parent
    save_dir = base_dir / "data" / "pdf"
    save_dir.mkdir(parents=True, exist_ok=True)

    rag_system: RAG = app.state.rag_system
    
    results = []
    documents_data = []
    per_file_metadata = []
    for file in files:
        # Accept only PDFs
        if not file.filename.lower().endswith(".pdf"):
            results.append({
                "filename": file.filename,
                "success": False,
                "error": "Only PDF files are supported."
            })
            continue

        saved_path = save_dir / file.filename
        try:
            with saved_path.open("wb") as out_file:
                shutil.copyfileobj(file.file, out_file)
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": f"Failed to save file: {e}"
            })
            file.file.close()
            continue
        finally:
            file.file.close()

        # Process PDF into chunks using existing PDFCleaner
        try:
            cleaner = PDFCleaner(min_block_size=100, max_block_size=500)
            chunks = cleaner.process_pdf(str(saved_path))

            # collect documents to ingest and per-file metadata for the final response
            documents_data.extend([
                {"file": file.filename, "chunk_id": str(i + 1), "content": chunk}
                for i, chunk in enumerate(chunks)
            ])
            per_file_metadata.append({
                "filename": file.filename,
                "chunks": len(chunks),
                "preview": chunks[:2]  # keep a 2-chunk preview consistently
            })

        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": f"Failed to process PDF: {e}"
            })
            continue

    # After processing all files, ingest chunks into Weaviate once
    try:
        if documents_data:
            rag_system.data_ingestion(documents_data)

        # If ingestion succeeds, mark all processed files as successful
        for meta in per_file_metadata:
            results.append({
                "filename": meta["filename"],
                "chunks": meta["chunks"],
                "preview": meta["preview"],
                "success": True
            })

    except Exception as e:
        # If ingestion fails, mark the processed files with an ingestion error
        for meta in per_file_metadata:
            results.append({
                "filename": meta["filename"],
                "success": False,
                "error": f"Failed to ingest data into Weaviate: {e}"
            })

    return {"uploads": results}

# Web UI route
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {"request": request})


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0", 
        port=8000, 
        reload=False
    )