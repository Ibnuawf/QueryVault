"""
Database Builder for RAG System
==============================
This module encapsulates the logic for constructing the ChromaDB vector store from raw Q&A data files.
It handles data ingestion, deduplication, chunking, embedding generation, and collection management.

Key Responsibilities:
- Parse and validate source JSON files.
- Deduplicate and chunk Q&A data for optimal retrieval.
- Generate embeddings and populate the ChromaDB collection.
- Provide robust error handling and logging for production reliability.

Usage:
Instantiate `DatabaseBuilder` and call `run()` to build or rebuild the vector database.
"""

# app/data_builder.py
import os
import json
import logging
import chromadb
from sentence_transformers import SentenceTransformer
from app.config import Settings

class DatabaseBuilder:
    """
    DatabaseBuilder
    ---------------
    Encapsulates the logic for building the ChromaDB vector store from Q&A data.
    Handles all steps from file parsing to embedding and collection creation.
    """
    def __init__(self, settings: Settings, model: SentenceTransformer):
        """
        Initialize the DatabaseBuilder with application settings and embedding model.
        Args:
            settings (Settings): Application configuration.
            model (SentenceTransformer): Embedding model for vectorization.
        """
        self.settings = settings
        self.embedding_model = model
        self.client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)

    def run(self) -> tuple[int, int]:
        """
        Build the Vector Database
        ------------------------
        Processes all data files, generates embeddings, and builds the ChromaDB collection.
        Returns:
            tuple[int, int]: (number of chunks, number of unique questions)
        """
        os.makedirs(self.settings.DATA_DIR, exist_ok=True)
        json_files = [f for f in os.listdir(self.settings.DATA_DIR) if f.endswith('.json')]
        if not json_files:
            logging.error("No JSON files found in 'data' directory. Cannot build database.")
            return 0, 0

        chunks, metadatas, seen_questions = self._process_files(json_files)
        if not chunks:
            logging.error("No valid data chunks were created from source files.")
            return 0, 0
            
        self._create_collection(chunks, metadatas)
        return len(chunks), len(seen_questions)

    def _process_files(self, json_files: list):
        """
        Process Source Files
        -------------------
        Iterates through all JSON files, extracts and chunks Q&A pairs, and deduplicates questions.
        Returns:
            tuple: (chunks, metadatas, seen_questions)
        """
        chunks, metadatas, seen_questions = [], [], set()
        for file_name in json_files:
            file_path = os.path.join(self.settings.DATA_DIR, file_name)
            logging.info(f"--> Processing file: {file_name}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for category_data in data.values():
                    for qna_item in category_data.get('questions', []):
                        q, a, url = qna_item.get('question'), qna_item.get('answer'), qna_item.get('url')
                        if q and a and q.strip().lower() not in seen_questions:
                            seen_questions.add(q.strip().lower())
                            words = a.strip().split()
                            for i in range(0, len(words), self.settings.CHUNK_SIZE_WORDS):
                                chunk_text = " ".join(words[i:i+self.settings.CHUNK_SIZE_WORDS])
                                document = f"Question: {q.strip()}\nAnswer: {chunk_text}"
                                chunks.append(document)
                                metadatas.append({'source': url or file_name})
            except Exception as e:
                logging.warning(f"Skipping file {file_name} due to error: {e}")
        return chunks, metadatas, seen_questions

    def _create_collection(self, chunks, metadatas):
        """
        Create or Rebuild ChromaDB Collection
        -------------------------------------
        Deletes any existing collection with the same name, then creates and populates a new one.
        Args:
            chunks (list): List of document chunks.
            metadatas (list): List of metadata dicts for each chunk.
        """
        collection_name = self.settings.COLLECTION_NAME
        if collection_name in [c.name for c in self.client.list_collections()]:
            self.client.delete_collection(name=collection_name)
        
        collection = self.client.create_collection(name=collection_name)
        
        logging.info(f"Embedding {len(chunks)} chunks...")
        embeddings = self.embedding_model.encode(chunks, show_progress_bar=True, normalize_embeddings=True)
        collection.add(
            ids=[f"id_{i}" for i in range(len(chunks))],
            embeddings=embeddings.tolist(),
            documents=chunks,
            metadatas=metadatas
        )