import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class EmbeddingManager:
    """Handles loading the embedding model and converting text into vectors."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # all-MiniLM-L6-v2 is a lightweight HuggingFace model — fast and good for semantic search
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the SentenceTransformer model from HuggingFace."""
        try:
            print(f"[Embedding] Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            dim = self.model.get_sentence_embedding_dimension()
            print(f"[Embedding] Model ready. Embedding dimension: {dim}")
        except Exception as e:
            print(f"[Error] Failed to load embedding model: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Convert a list of text strings into embedding vectors.
        Returns a numpy array of shape (len(texts), embedding_dim).
        """
        if not self.model:
            raise ValueError("Model not loaded. Something went wrong during init.")

        print(f"[Embedding] Generating embeddings for {len(texts)} text(s)...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"[Embedding] Done. Shape: {embeddings.shape}")
        return np.array(embeddings)


def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Split documents into smaller chunks so the LLM gets focused, relevant context.
    chunk_overlap keeps a bit of context between consecutive chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        # try to split on paragraph breaks first, then lines, then words
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(documents)
    print(f"[Splitter] {len(documents)} document(s) → {len(chunks)} chunks")
    return chunks
