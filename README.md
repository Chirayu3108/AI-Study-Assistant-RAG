# 📚 AI Study Assistant — RAG-Powered Document Q&A

An AI-powered study assistant that lets you upload your own study materials and ask questions about them. It uses Retrieval-Augmented Generation (RAG) to find the most relevant parts of your documents and then uses an LLM to give you a clear, concise answer — like having a tutor that has read everything you gave it.

---

## What it does

- Upload any study material — PDF, TXT, CSV, DOCX, Excel, or JSON
- The documents are chunked, embedded, and stored in a local vector database (ChromaDB)
- When you ask a question, the app finds the most relevant chunks using semantic search and sends them to a Groq-hosted LLaMA model to generate an answer
- Shows you the source file and page number for every answer so you can verify it yourself
- Fully containerized with Docker so it runs the same everywhere

---

## Architecture

```
User uploads file
        ↓
  Document Loader        ← src/data_loader.py
  (PDF/TXT/CSV/DOCX...)
        ↓
  Text Chunking          ← src/embedding.py
  (RecursiveCharacterTextSplitter)
        ↓
  Embedding              ← src/embedding.py
  (all-MiniLM-L6-v2 via SentenceTransformers)
        ↓
  Vector Store           ← src/vectorstore.py
  (ChromaDB, persisted to disk)
        ↓
  User asks a question
        ↓
  Semantic Retrieval     ← src/search.py
  (cosine similarity search)
        ↓
  LLM Answer Generation  ← src/llm.py
  (Groq — LLaMA 3.1 8B Instant)
        ↓
  Streamlit Frontend     ← app.py
```

---

## Tech Stack

| Component | Tool |
|---|---|
| Frontend | Streamlit |
| Document Loaders | LangChain Community |
| Embeddings | SentenceTransformers (`all-MiniLM-L6-v2`) |
| Vector Database | ChromaDB |
| LLM | Groq (LLaMA 3.1 8B Instant) |
| Containerization | Docker + Docker Compose |

---

## Project Structure

```
AI-Study-Assistant-RAG/
│
├── src/
│   ├── data_loader.py       # Loads PDF, TXT, CSV, DOCX, XLSX, JSON files
│   ├── embedding.py         # EmbeddingManager + document chunking
│   ├── vectorstore.py       # ChromaDB wrapper for storing/querying embeddings
│   ├── search.py            # RAGRetriever — semantic search over the vector store
│   ├── llm.py               # Groq LLM setup + answer generation with study prompt
│   └── rag_pipeline.py      # Orchestrates the full pipeline end to end
│
├── data/
│   ├── pdf_files/           # Sample PDF study materials
│   ├── text_files/          # Sample TXT files
│   └── vector_store/        # ChromaDB persists here
│
├── notebook/
│   └── pdf_loader.ipynb     # Original experimental notebook
│
├── app.py                   # Streamlit web app
├── Dockerfile               # Container definition
├── docker-compose.yml       # One-command deploy
├── requirements.txt         # Python dependencies
└── .env                     # API keys (not committed to git)
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- A [Groq API key](https://console.groq.com) (free)
- Docker (optional, for containerized run)

### 1. Clone the repo

```bash
git clone https://github.com/your-username/AI-Study-Assistant-RAG.git
cd AI-Study-Assistant-RAG
```

### 2. Set up your API key

Create a `.env` file in the root:

```
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Running with Docker

```bash
# Build and start the container
docker compose up --build

# Stop it
docker compose down
```

The app will be available at [http://localhost:8501](http://localhost:8501).

The `data/` folder is mounted as a volume so your indexed documents survive container restarts.

---

## How to use it

1. **Upload files** — use the sidebar to upload your notes, textbooks, or any study material
2. **Click "Index Documents"** — this processes and stores them in the vector database
3. **Ask a question** — type your question in the chat box and hit Enter
4. The assistant will answer based only on what's in your documents, and show you exactly where the information came from

---

## Supported File Types

| Format | Extension |
|---|---|
| PDF | `.pdf` |
| Plain Text | `.txt` |
| CSV | `.csv` |
| Word Document | `.docx`, `.doc` |
| Excel Spreadsheet | `.xlsx`, `.xls` |
| JSON | `.json` |

---

## Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Your Groq API key for LLaMA inference |

---

## Notes

- The vector store is persisted to `data/vector_store/` so you don't have to re-index every time you restart the app
- The embedding model (`all-MiniLM-L6-v2`) runs locally — no API call needed for embeddings
- Answers are grounded in your documents — the LLM is instructed not to make things up
