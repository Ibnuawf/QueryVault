# QueryVault

A robust, production-ready Retrieval-Augmented Generation (RAG) application leveraging FastAPI, Typer, ChromaDB, and Google Gemini. Designed for scalable, maintainable, and secure knowledge retrieval and question answering.

---

## 🚀 Features

- **FastAPI Backend:** High-performance, asynchronous API server for real-time Q&A.
- **Modular Architecture:** Clean separation of web, business logic, data pipeline, and configuration.
- **Typer CLI:** Powerful command-line interface for database management and server operations.
- **ChromaDB Integration:** Efficient vector database for semantic search and retrieval.
- **Google Gemini LLM:** State-of-the-art language model for accurate, context-aware answers.
- **Pydantic Settings:** Type-safe, environment-driven configuration management.
- **Production Hardening:** In-memory rate limiting, robust error handling, and secure secret management.
- **Modern Frontend:** Responsive, minimal HTML interface for end users.

---

## 🗂️ Project Structure

```
QueryVault/
├── .env                  # Environment variables (API keys, secrets)
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
│
├── app/                  # Main application package
│   ├── main.py           # FastAPI app, endpoints, startup logic
│   ├── cli.py            # Typer CLI commands
│   ├── config.py         # Pydantic settings/configuration
│   ├── core.py           # Core RAG pipeline logic
│   ├── data_builder.py   # Database ingestion/building logic
│   ├── dependencies.py   # Shared dependencies (e.g., rate limiter)
│   └── templates/
│       └── index.html    # HTML frontend
│
└── data/                 # Input Q&A JSON files
    └── my_dataset.json
```

---

## ⚙️ Setup & Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Ibnuawf/QueryVault
   cd QueryVault
   ```

2. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   # On Unix/macOS:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```

3. **Configure Environment Variables**
   Create a `.env` file in the project root:

   ```env
   GEMINI_API_KEY=your-google-ai-api-key
   ```

4. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

5. **Add Your Data**
   Place your Q&A JSON files (e.g., `my_dataset.json`) in the `data/` directory.

---

## 🏗️ Building & Running

### 1. Build the Vector Database

Process your data and create the ChromaDB vector store:

```bash
python -m app.cli build-db
```

### 2. Start the Web Application

Launch the FastAPI server:

```bash
python -m app.cli run-app
```

- Access the app at: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- API documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 📝 Best Practices & Notes

- **Security:** Never commit real API keys or secrets to public repositories. Use `.env` for sensitive config.
- **Data Format:** Ensure your Q&A JSON files are well-structured and validated before ingestion.
- **Extensibility:** The modular design allows easy extension for new data sources, models, or endpoints.
- **Error Handling:** All major operations include robust error handling and logging for production reliability.
- **Rate Limiting:** The API enforces per-client rate limits to prevent abuse.

---

## 📚 License & Attribution

This project is provided for educational and research purposes. For commercial use, review all dependencies and comply with their respective licenses.

---

## 🙏 Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/)
- [ChromaDB](https://www.trychroma.com/)
- [Google Gemini](https://ai.google.dev/)
- [Sentence Transformers](https://www.sbert.net/)
- [Typer](https://typer.tiangolo.com/)
- [Pydantic](https://docs.pydantic.dev/)
