from typing import List, Dict, Any
from src.embedding import EmbeddingManager
from src.vectorstore import VectorStore


class RAGRetriever:
    """Retrieves the most relevant document chunks from the vector store for a given query."""

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Embed the query, then pull the top_k closest chunks from ChromaDB.

        Args:
            query: the student's question
            top_k: how many chunks to return

        Returns:
            List of dicts with content, metadata, similarity score, and rank.
        """
        print(f"[Retriever] Query: '{query}'")

        # convert the query to an embedding vector
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        # ChromaDB errors if n_results > number of documents in the collection
        # so we cap it to whatever is actually stored
        doc_count = self.vector_store.count()
        if doc_count == 0:
            print("[Retriever] Collection is empty — nothing to search.")
            return []
        n_results = min(top_k, doc_count)

        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
            )
        except Exception as e:
            print(f"[Error] Retrieval failed: {e}")
            return []

        retrieved = []

        if results["documents"] and results["documents"][0]:
            for doc_id, document, metadata, distance in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                # ChromaDB cosine distance ranges 0–2, so similarity = 1 - distance
                # ranges from -1 to 1. We return ALL results and let the LLM judge relevance.
                similarity = 1 - distance

                retrieved.append({
                    "id": doc_id,
                    "content": document,
                    "metadata": metadata,
                    "similarity_score": round(similarity, 4),
                    "rank": len(retrieved) + 1,
                })

        print(f"[Retriever] Found {len(retrieved)} chunk(s)")
        return retrieved
