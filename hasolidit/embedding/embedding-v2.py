import json
from typing import List, Dict
import os
from langchain_community.document_loaders import JSONLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


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

def chunk_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Split documents into smaller chunks for more granular embedding.
    
    :param documents: List of original Langchain Documents
    :param chunk_size: Maximum number of characters in each chunk
    :param chunk_overlap: Number of characters to overlap between chunks
    :return: List of chunked documents
    """
    # Create a text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        add_start_index=True
    )
    
    # Split documents
    chunked_documents = []
    for doc in documents:
        # Create chunks while preserving original metadata
        chunks = text_splitter.split_text(doc.page_content)
        
        # Create new documents for each chunk
        for i, chunk in enumerate(chunks):
            # Create a copy of metadata for each chunk
            chunk_metadata = doc.metadata.copy()
            # Add chunk-specific metadata
            chunk_metadata['chunk_id'] = i
            chunk_metadata['original_doc_url'] = chunk_metadata.get('url', '')
            
            chunked_doc = Document(page_content=chunk, metadata=chunk_metadata)
            chunked_documents.append(chunked_doc)
    
    return chunked_documents

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

def search_faiss_socuments(query: str, index: str, embedding_model: str) -> List[Document]:
    embeddings = OllamaEmbeddings(model=embedding_model)
    vector_store = FAISS.load_local(
        index, embeddings, allow_dangerous_deserialization=True
    )
    docs = vector_store.similarity_search(query)
    return docs


def main():
    # Configuration - replace these with your specific paths and settings
    JSON_FILE_PATH = 'hasolidit_articles.json'
    EMBEDDING_MODEL = 'llama3'  # Example Ollama embedding model
    INDEX_NAME = ""
    PERSIST_DIRECTORY = './faiss_index'

    CHUNK_SIZE = 1000  # Maximum characters per chunk
    CHUNK_OVERLAP = 200
    
    # Ensure persist directory exists
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
    
    # Load documents from JSON
    original_documents = load_json_documents(JSON_FILE_PATH)

    chunked_documents = chunk_documents(
        original_documents, 
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP
    )

    for doc in chunked_documents:
        print(doc)
    exit(0)
    # Create and persist FAISS index
    vectorstore = create_faiss_index(chunked_documents, EMBEDDING_MODEL, PERSIST_DIRECTORY)
    
    # Optional: Demonstrate retrieval
    query = "Your search query here"
    retrieved_docs = search_faiss_socuments(query, "faiss_index", EMBEDDING_MODEL)
    
    print("\nRetrieved Documents:")
    for doc in retrieved_docs:
        print(f"Score: {doc.metadata}")
        print(f"Content: {doc.page_content[:200]}...\n")

if __name__ == "__main__":
    main()