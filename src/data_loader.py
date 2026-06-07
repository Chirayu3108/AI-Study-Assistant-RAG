from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader

def load_all_documents(data_dir: str) -> List[Any]:
    '''
    Load all supported files from the data directory and conver to LangChain document structure
    Supported: CSV, PDF, TXT, EXCEL, WORD, JSON

    '''

    # Use project root data folder
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data Path: {data_path}")
    documents =[]

    #PDF files
    pdf_files = list(data_path.glob('**/*.pdf'))
   # print(f'[DEBUG] Found {len(pdf_files)} PDF files: {str(f) for f in pdf_files}')
    for pdf_file in pdf_files:
        print(f"[DEBUG] Loading PDF: ") 
        try:
            loader = PyPDFLoader(str(pdf_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} PDF docs from {pdf_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[Error] Failed to load PDF {pdf_file}: {e}")
            

    # TEXT files
    #PDF files
    text_files = list(data_path.glob('**/*.txt'))
   # print(f'[DEBUG] Found {len(pdf_files)} PDF files: {str(f) for f in pdf_files}')
    for text_file in text_files:
        print(f"[DEBUG] Loading Text File: ") 
        try:
            loader = TextLoader(str(text_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} Text file docs from {text_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[Error] Failed to load text file {text_file}: {e}")

    return documents