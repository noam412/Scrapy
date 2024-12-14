import json
from typing import List, Dict
import os
from langchain_community.document_loaders import JSONLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

def load_json_documents(file_path: str) -> List[Document]:
    """
    Load documents from a JSON file, converting each entry to a Langchain Document.
    
    :param file_path: Path to the JSON file containing documents
    :return: List of Langchain Documents
    """
    documents = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    for item in data:
        # Create metadata dictionary with all available fields
        metadata = {
            'url': item.get('url', ''),
            'article_number': item.get('article_number', ''),
            'title': item.get('title', ''),
        }
        
        # Combine title and body for full text
        page_content = f"{item.get('title', '')} {item.get('body', '')}"
        
        # Create Langchain Document
        doc = Document(page_content=page_content, metadata=metadata)
        documents.append(doc)
    
    return documents

def create_faiss_index(documents: List[Document], embedding_model: str, persist_directory: str):
    """
    Create a FAISS vector store from documents and persist it locally.
    
    :param documents: List of Langchain Documents
    :param embedding_model: Ollama embedding model to use
    :param persist_directory: Directory to save the FAISS index
    """
    # Create Ollama embeddings
    embeddings = OllamaEmbeddings(model=embedding_model)
    
    # Create FAISS vector store
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # Persist the index locally
    vectorstore.save_local(persist_directory)
    print(f"FAISS index saved to {persist_directory}")
    
    return vectorstore

def main():
    # Configuration - replace these with your specific paths and settings
    JSON_FILE_PATH = 'hasolidit_articles.json'
    EMBEDDING_MODEL = 'llama3'  # Example Ollama embedding model
    PERSIST_DIRECTORY = './faiss_index'
    
    # Ensure persist directory exists
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
    
    # Load documents from JSON
    documents = load_json_documents(JSON_FILE_PATH)

    # Create and persist FAISS index
    vectorstore = create_faiss_index(documents, EMBEDDING_MODEL, PERSIST_DIRECTORY)
    
    # Optional: Demonstrate retrieval
    query = "Your search query here"
    retrieved_docs = vectorstore.similarity_search(query, k=3)
    
    print("\nRetrieved Documents:")
    for doc in retrieved_docs:
        print(f"Score: {doc.metadata}")
        print(f"Content: {doc.page_content[:200]}...\n")

if __name__ == "__main__":
    main()