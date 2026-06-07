import os
import uuid
import numpy as np
from typing import List, Any
import chromadb
from langchain_core.documents import Document


class VectorStore:
    """Manages storing and querying document embeddings using ChromaDB."""

    def __init__(self, collection_name: str = "study_documents", persist_directory: str = "data/vector_store"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        """Set up the ChromaDB persistent client and get or create the collection."""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)

            # PersistentClient saves the vector store to disk between sessions
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Study document embeddings for RAG"}
            )

            print(f"[VectorStore] Collection '{self.collection_name}' ready.")
            print(f"[VectorStore] Documents already stored: {self.collection.count()}")

        except Exception as e:
            print(f"[Error] VectorStore init failed: {e}")
            raise

    def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        """
        Add documents and their embeddings to ChromaDB.
        Each document gets a unique ID so we avoid duplicates on re-runs.
        """
        if len(documents) != len(embeddings):
            raise ValueError("Mismatch: number of documents must equal number of embeddings.")

        print(f"[VectorStore] Adding {len(documents)} documents...")

        ids, metadatas, texts, embedding_list = [], [], [], []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            # flatten metadata — ChromaDB only accepts str/int/float/bool values
            metadata = {k: str(v) for k, v in doc.metadata.items()}
            metadata["content_length"] = len(doc.page_content)
            metadatas.append(metadata)

            texts.append(doc.page_content)
            embedding_list.append(embedding.tolist())

        try:
            self.collection.add(
                ids=ids,
                embeddings=embedding_list,
                metadatas=metadatas,
                documents=texts,
            )
            print(f"[VectorStore] Stored. Total in collection: {self.collection.count()}")
        except Exception as e:
            print(f"[Error] Failed to add documents: {e}")
            raise

    def clear(self):
        """Delete and recreate the collection — useful when re-indexing new files."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Study document embeddings for RAG"}
        )
        print(f"[VectorStore] Collection cleared and reset.")

    def count(self) -> int:
        return self.collection.count()
