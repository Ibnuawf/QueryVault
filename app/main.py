"""
Main API Entrypoint for Professional RAG System
===============================================
This module defines the FastAPI application, API endpoints, and startup logic for the Retrieval-Augmented Generation (RAG) system.
It orchestrates model loading, database connection, and request handling for both web and API clients.

Key Responsibilities:
- Initialize and configure the FastAPI app and all required resources.
- Define API endpoints for user interaction and knowledge retrieval.
- Serve the main HTML frontend and provide streaming answers via the RAG pipeline.
- Ensure robust error handling and production reliability.

Usage:
This file is the main entrypoint for the web server. Run via CLI or as a module.
"""

import logging
import os
import torch
from fastapi import FastAPI, Depends, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# Local Imports
from app.config import Settings, get_settings
from app.dependencies import rate_limiter
from app.core import generate_answer_stream

# Import AI/DB libraries for type hinting and setup
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import chromadb

# Suppress PyTorch and other loggers if needed
# Ensures deterministic, single-threaded inference for production
torch.set_num_threads(1)

# --- FastAPI App Initialization ---
app = FastAPI(title="Professional RAG API", version="2.0.0")

# Setup for serving HTML template
templates = Jinja2Templates(directory="app/templates")

@app.on_event("startup")
async def startup_event():
    """
    Application Startup Task
    -----------------------
    Loads all required models and connects to the ChromaDB collection on server startup.
    Ensures that all resources are available before serving requests.
    """
    logging.info("Loading application resources...")
    settings = get_settings()
    app.state.settings = settings
    app.state.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
    
    genai.configure(api_key=settings.GEMINI_API_KEY)
    app.state.llm = genai.GenerativeModel(settings.GEMINI_MODEL_NAME)

    try:
        client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        app.state.chroma_collection = client.get_collection(name=settings.COLLECTION_NAME)
        if app.state.chroma_collection.count() == 0:
            raise ValueError(f"Collection '{settings.COLLECTION_NAME}' is empty.")
    except Exception as e:
        logging.critical(f"FATAL: Failed to connect to DB. Did you run `build-db`? Error: {e}")
        raise SystemExit(1)
    logging.info("Resources loaded successfully.")

# --- API Data Models ---
class AskRequest(BaseModel):
    """
    Request Model for /ask Endpoint
    ------------------------------
    Defines the expected structure for user queries submitted to the API.
    """
    query: str = Field(..., min_length=3, max_length=200, description="User's question.")

# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home(request: Request):
    """
    Home Page Endpoint
    -----------------
    Serves the main HTML frontend for the RAG assistant.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask", dependencies=[Depends(rate_limiter)])
async def ask(ask_request: AskRequest, request: Request) -> StreamingResponse:
    """
    RAG Answer Endpoint
    ------------------
    Accepts a user query, performs retrieval-augmented generation, and streams the response.
    """
    pipeline_generator = generate_answer_stream(
        query=ask_request.query,
        settings=request.app.state.settings,
        collection=request.app.state.chroma_collection,
        embedding_model=request.app.state.embedding_model,
        llm=request.app.state.llm
    )
    return StreamingResponse(pipeline_generator, media_type="text/event-stream")