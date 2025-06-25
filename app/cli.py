"""
CLI Entrypoint for Professional RAG System
=========================================
This module provides a robust command-line interface for managing the lifecycle of the Retrieval-Augmented Generation (RAG) application.
It supports database building, server startup, and other operational tasks, ensuring a seamless workflow for developers and operators.

Key Responsibilities:
- Build and manage the ChromaDB vector database from source data.
- Launch the FastAPI web server for the RAG API.
- Provide clear, user-friendly CLI feedback and error handling.

Usage:
Run this module directly or use its commands via `python -m app.cli ...`.
"""

# app/cli.py
import logging
import typer
import uvicorn
from sentence_transformers import SentenceTransformer

# Local Imports
from app.main import app  # Import the FastAPI app object
from app.config import get_settings
from app.data_builder import DatabaseBuilder

# Setup CLI
cli = typer.Typer(help="A professional RAG application CLI.")

@cli.command()
def build_db():
    """
    Build Vector Database
    --------------------
    Processes all source JSON files, generates embeddings, and builds the ChromaDB vector store.
    Ensures deduplication, chunking, and robust error handling for production use.
    """
    typer.secho("Building vector database...", fg=typer.colors.CYAN)
    settings = get_settings()
    model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
    builder = DatabaseBuilder(settings, model)
    num_chunks, num_questions = builder.run()
    if num_chunks > 0:
        typer.secho(f"✅ Success! Built DB '{settings.COLLECTION_NAME}' with {num_chunks} chunks from {num_questions} unique questions.", fg=typer.colors.GREEN)

@cli.command()
def run_app(host: str = "0.0.0.0", port: int = 8000):
    """
    Start FastAPI Server
    -------------------
    Launches the Uvicorn server to serve the RAG API, making the system accessible via HTTP.
    Provides clear startup feedback and API documentation URL.
    """
    typer.secho(f"🚀 Starting RAG server at http://{host}:{port}", fg=typer.colors.BRIGHT_BLUE)
    typer.secho("   Access API docs at http://{host}:{port}/docs", fg=typer.colors.BLUE)
    uvicorn.run(app, host=host, port=port)

# This check allows the file to be run directly
if __name__ == "__main__":
    cli()