from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    JSONLoader,
)
from langchain_community.document_loaders.excel import UnstructuredExcelLoader


def load_documents_from_path(file_path: str) -> List[Any]:
    """
    Load a single file based on its extension.
    Returns a list of LangChain Document objects.
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    documents = []

    try:
        # pick the right loader based on file type
        if ext == ".pdf":
            loader = PyMuPDFLoader(str(path))
        elif ext == ".txt":
            loader = TextLoader(str(path))
        elif ext == ".csv":
            loader = CSVLoader(str(path))
        elif ext in (".docx", ".doc"):
            loader = Docx2txtLoader(str(path))
        elif ext in (".xlsx", ".xls"):
            loader = UnstructuredExcelLoader(str(path))
        elif ext == ".json":
            # jq_schema="." means load the entire JSON content as text
            loader = JSONLoader(str(path), jq_schema=".", text_content=False)
        else:
            print(f"[Warning] Unsupported file type: {ext} — skipping {path.name}")
            return []

        loaded = loader.load()

        # tag each doc with source file info so we know where it came from
        for doc in loaded:
            doc.metadata["source_file"] = path.name
            doc.metadata["file_type"] = ext.lstrip(".")

        documents.extend(loaded)
        print(f"[Loaded] {path.name} → {len(loaded)} document(s)")

    except Exception as e:
        print(f"[Error] Could not load {path.name}: {e}")

    return documents


def load_all_documents(data_dir: str) -> List[Any]:
    """
    Recursively scan a directory and load all supported files.
    Supported: PDF, TXT, CSV, DOCX, XLSX, JSON
    """
    data_path = Path(data_dir).resolve()
    print(f"[Loader] Scanning: {data_path}")

    # all supported extensions
    supported_extensions = ["*.pdf", "*.txt", "*.csv", "*.docx", "*.doc", "*.xlsx", "*.xls", "*.json"]

    all_documents = []

    for pattern in supported_extensions:
        for file_path in data_path.rglob(pattern):
            docs = load_documents_from_path(str(file_path))
            all_documents.extend(docs)

    print(f"[Loader] Total documents loaded: {len(all_documents)}")
    return all_documents
