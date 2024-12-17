import json
import os
from typing import List, Any
from dotenv import load_dotenv
from langchain_community.document_loaders import JSONLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()


def load_json_documents(file_path: str) -> List[Document]:
    """
    Load documents from a JSON file, converting each entry to a Langchain Document.
    """
    documents = []
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    print(f"Loaded {len(data)} articles from JSON.")
    
    for item in data:
        # Create metadata dictionary with all available fields
        metadata = {
            'url': item.get('url', ''),
            'article_number': item.get('article_number', ''),
            'title': item.get('title', ''),
        }
        # Combine title and body for full text
        page_content = f"{item.get('title', '')} {item.get('body', '')}"
        
        # Skip empty documents
        if not page_content.strip():
            print(f"Skipping empty document for URL: {metadata['url']}")
            continue
        
        # Create Langchain Document
        doc = Document(page_content=page_content, metadata=metadata)
        documents.append(doc)
    return documents


def chunk_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Split documents into smaller chunks for more granular embedding.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        add_start_index=True
    )
    chunked_documents = []
    for doc in documents:
        chunks = text_splitter.split_text(doc.page_content)
        print(f"Document: {doc.metadata['title']} - Chunks created: {len(chunks)}")
        for i, chunk in enumerate(chunks):
            chunk_metadata = doc.metadata.copy()
            chunk_metadata['chunk_id'] = i
            chunk_metadata['original_doc_url'] = chunk_metadata.get('url', '')
            chunked_doc = Document(page_content=chunk, metadata=chunk_metadata)
            chunked_documents.append(chunked_doc)
    return chunked_documents

def create_faiss_index(documents: List[Document], embeddings: Any, persist_directory: str) -> None:
    """
    Create a FAISS vector store from documents and persist it locally.
    """
    
    # Create FAISS vector store
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # Persist the index locally
    vectorstore.save_local(persist_directory)
    print(f"FAISS index saved to {persist_directory}")

def search_faiss_socuments(query: str, index: str, embeddings: Any) -> List[Document]:
    """ Query Documents from Faiss Vector DB """
    vector_store = FAISS.load_local(
        index, embeddings, allow_dangerous_deserialization=True
    )
    docs = vector_store.similarity_search(query)
    return docs


def main():
    # Configuration
    JSON_FILE_PATH = 'hasolidit_articles.json.json'  # Path to the JSON file
    EMBEDDING_MODEL = 'text-embedding-3-large'  # Example Ollama embedding model
    INDEX_NAME = "faiss_index"
    PERSIST_DIRECTORY = f'./{INDEX_NAME}'

    CHUNK_SIZE = 1000  # Maximum characters per chunk
    CHUNK_OVERLAP = 200

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    
    # Ensure persist directory exists
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
    
    # Step 1: Load JSON documents
    print("\nLoading documents...")
    original_documents = load_json_documents(JSON_FILE_PATH)

    chunked_documents = chunk_documents(
        original_documents, 
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP
    )

    # Create and persist FAISS index
    vectorstore = create_faiss_index(chunked_documents, embeddings, PERSIST_DIRECTORY)

if __name__ == "__main__":
    main()