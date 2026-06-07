from typing import List, Dict, Any
from src.data_loader import load_documents_from_path, load_all_documents
from src.embedding import EmbeddingManager, split_documents
from src.vectorstore import VectorStore
from src.search import RAGRetriever
from src.llm import get_llm, generate_answer


class RAGPipeline:
    """
    Ties the full pipeline together:
    Load → Chunk → Embed → Store → Retrieve → Generate
    """

    def __init__(self, persist_directory: str = "data/vector_store"):
        print("[Pipeline] Initializing RAG pipeline...")
        self.embedding_manager = EmbeddingManager()
        self.vector_store = VectorStore(persist_directory=persist_directory)
        self.retriever = RAGRetriever(self.vector_store, self.embedding_manager)
        self.llm = get_llm()
        print("[Pipeline] Ready.")

    def ingest_directory(self, data_dir: str, clear_existing: bool = False):
        """
        Load all supported files from a directory, chunk them, embed them, and store in ChromaDB.
        Set clear_existing=True to re-index from scratch.
        """
        if clear_existing:
            print("[Pipeline] Clearing existing vector store...")
            self.vector_store.clear()

        documents = load_all_documents(data_dir)
        if not documents:
            print("[Pipeline] No documents found.")
            return 0

        chunks = split_documents(documents)
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embedding_manager.generate_embeddings(texts)
        self.vector_store.add_documents(chunks, embeddings)

        print(f"[Pipeline] Ingested {len(chunks)} chunks from {data_dir}")
        return len(chunks)

    def ingest_file(self, file_path: str):
        """
        Load, chunk, embed, and store a single uploaded file.
        Used by the frontend when a user uploads a document.
        """
        documents = load_documents_from_path(file_path)
        if not documents:
            print(f"[Pipeline] Could not load file: {file_path}")
            return 0

        chunks = split_documents(documents)
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embedding_manager.generate_embeddings(texts)
        self.vector_store.add_documents(chunks, embeddings)

        print(f"[Pipeline] Ingested {len(chunks)} chunks from {file_path}")
        return len(chunks)

    def ask(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Run the full query pipeline:
        Retrieve relevant chunks → build prompt → generate answer.
        """
        retrieved = self.retriever.retrieve(query, top_k=top_k)
        result = generate_answer(query, retrieved, self.llm)
        return result

    def get_document_count(self) -> int:
        return self.vector_store.count()
