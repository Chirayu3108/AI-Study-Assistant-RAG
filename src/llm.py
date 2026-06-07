import os
from typing import List, Dict, Any
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


def get_llm(model_name: str = "llama-3.1-8b-instant", temperature: float = 0.2) -> ChatGroq:
    """Initialize and return the Groq LLM."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment. Add it to your .env file.")

    return ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        max_tokens=1024,
    )


def build_prompt(query: str, context: str) -> str:
    """
    Build the study assistant prompt.
    The context comes from the retrieved document chunks.
    """
    return f"""You are an AI Study Assistant helping a student understand their study material.
Use ONLY the context provided below to answer the question. If the context does not contain enough information, say so honestly — do not make things up.

Keep your answer clear, structured, and easy to understand. Use simple language and examples where helpful.
If explaining a concept, break it down step by step.

Context from study material:
{context}

Student's Question: {query}

Answer:"""


def generate_answer(query: str, retrieved_docs: List[Dict[str, Any]], llm: ChatGroq) -> Dict[str, Any]:
    """
    Take retrieved chunks and the query, build a prompt, and get the LLM to answer.

    Returns a dict with:
        - answer: the LLM's response text
        - sources: list of source file + page info
        - confidence: the highest similarity score among retrieved docs
    """
    if not retrieved_docs:
        return {
            "answer": "I couldn't find any relevant information in the uploaded documents to answer your question.",
            "sources": [],
            "confidence": 0.0,
        }

    # join all chunks into one context block
    context = "\n\n".join([doc["content"] for doc in retrieved_docs])

    prompt = build_prompt(query, context)

    response = llm.invoke(prompt)

    # collect source info for citation display
    sources = []
    for doc in retrieved_docs:
        source_info = {
            "file": doc["metadata"].get("source_file", doc["metadata"].get("source", "unknown")),
            "page": doc["metadata"].get("page", "N/A"),
            "score": round(doc["similarity_score"], 3),
            "preview": doc["content"][:200] + "...",
        }
        sources.append(source_info)

    # confidence = best matching chunk's similarity score
    confidence = max(doc["similarity_score"] for doc in retrieved_docs)

    return {
        "answer": response.content,
        "sources": sources,
        "confidence": round(confidence, 3),
    }
