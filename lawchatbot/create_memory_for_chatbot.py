import os
import json
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

PDF_PATH = "data/"
JSONL_PATH = "data/data.jsonl"
DB_FAISS_PATH = "vectorstore/db_faiss"

def load_pdf_files(data):
    loader = DirectoryLoader(data, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} PDF documents.")
    return documents

def load_jsonl_file(jsonl_path):
    documents = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            query = data.get("query", "").strip()
            response = data.get("response", "").strip()
            if query or response:
                page_content = f"Query: {query}\nResponse: {response}" if query and response else query or response
                documents.append(
                    Document(page_content=page_content, metadata={"id": data.get("id", ""), "source": "JSONL"})
                )
    print(f"Loaded {len(documents)} JSONL entries.")
    return documents

def create_chunks(extracted_data, chunk_size=500, chunk_overlap=50):
    chunks = []
    for doc in extracted_data:
        text = doc.page_content if hasattr(doc, 'page_content') else None
        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
        if text:
            for i in range(0, len(text), chunk_size - chunk_overlap):
                chunk_text = text[i:i + chunk_size]
                chunks.append(Document(page_content=chunk_text, metadata=metadata))
    print(f"Total chunks created: {len(chunks)}")
    return chunks

def main():
    pdf_documents = load_pdf_files(data=PDF_PATH)
    jsonl_documents = load_jsonl_file(JSONL_PATH)
    combined_data = pdf_documents + jsonl_documents

    text_chunks = create_chunks(extracted_data=combined_data)
    print(f"Text chunks created: {len(text_chunks)}")

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    try:
        vectorstore = FAISS.from_documents(text_chunks, embedding_model)
        vectorstore.save_local(DB_FAISS_PATH)
        print(f"Vector store saved to {DB_FAISS_PATH}")
    except Exception as e:
        print(f"Error saving vector store: {e}")

if __name__ == "__main__":
    main()
