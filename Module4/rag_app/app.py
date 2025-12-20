from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from contextlib import asynccontextmanager
from rag_system import RAG

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

# Web UI route
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {"request": request})
