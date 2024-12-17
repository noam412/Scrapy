#!/usr/bin/env python3
import os
import time
import logging
import sys
import signal
import concurrent.futures

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Set up logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(sys.stdout),
                        logging.FileHandler('chatbot.log')
                    ])
logger = logging.getLogger(__name__)

# Constants
EMBEDDING_MODEL = "llama3"
LLM_MODEL = "llama3"
PERSIST_DIRECTORY = "./faiss_index"
TARGET_SOURCE_CHUNKS = 4
TIMEOUT = 60  # 60 seconds timeout for response generation

def timeout_handler(signum, frame):
    raise TimeoutError("Response generation timed out")

def create_financial_advice_chain():
    """
    Create a chain that retrieves documents and generates financial advice.
    """
    try:
        # Initialize embeddings and vector store
        logger.info(f"Initializing OllamaEmbeddings with model: {EMBEDDING_MODEL}")
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        
        logger.info(f"Loading FAISS index from {PERSIST_DIRECTORY}")
        vectorstore = FAISS.load_local(PERSIST_DIRECTORY, embeddings, allow_dangerous_deserialization=True)
        
        # Initialize language model
        logger.info(f"Initializing ChatOllama with model: {LLM_MODEL}")
        llm = ChatOllama(
            model=LLM_MODEL, 
            temperature=0.7, 
            num_predict=500,  # Limit response length
            stop=["</response>"]  # Optional stop sequence
        )
        
        # Define a simple, flexible prompt template
        prompt_template = PromptTemplate.from_template(
            """Provide a helpful response based on the context and question.
            
            Context:
            {context}
            
            Question: {question}
            
            Response:"""
        )
        
        # Create the retrieval chain
        retriever = vectorstore.as_retriever(search_kwargs={"k": TARGET_SOURCE_CHUNKS})
        
        def format_docs(docs):
            if not docs:
                logger.warning("No documents retrieved")
                return "No relevant documents found."
            logger.info(f"Retrieved {len(docs)} documents")
            return "\n\n".join(doc.page_content for doc in docs)
        
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
        )
        
        return chain
    
    except Exception as e:
        logger.error(f"Error creating advice chain: {e}")
        raise

def generate_response_with_timeout(chain, query):
    """
    Generate a response with a timeout mechanism
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(chain.invoke, query)
        try:
            # Wait for the future to complete
            result = future.result(timeout=TIMEOUT)
            return result
        except concurrent.futures.TimeoutError:
            logger.error("Response generation timed out")
            return "I apologize, but I couldn't generate a response in time. Please try again."
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"An error occurred: {str(e)}"

def main():
    print("Welcome to the Financial Advisory Chatbot. Type 'exit' to quit.\n")
    
    try:
        # Create the advice chain once to avoid reloading for each query
        advice_chain = create_financial_advice_chain()
    except Exception as e:
        print(f"Failed to initialize chatbot: {e}")
        return

    while True:
        # Get user input
        try:
            query = input("Enter your financial query: ")
        except KeyboardInterrupt:
            print("\nInterrupted. Exiting...")
            break
        
        if query.lower() == "exit":
            print("\nExiting chatbot. Goodbye!")
            break
        
        if not query.strip():
            print("Empty query. Please ask something meaningful.")
            continue

        print("\nAnalyzing documents and generating advice...")
        
        try:
            # Generate financial advice
            start_time = time.time()
            
            # Generate response with timeout
            advice = generate_response_with_timeout(advice_chain, query)
            
            print("\n--- Financial Advice ---")
            print(advice)
            print("\n--- End of Advice ---")
            
            logger.info(f"Query processed in {time.time() - start_time:.2f} seconds")
        
        except Exception as e:
            print(f"Error generating advice: {e}")
            logger.error(f"Advice generation error: {e}")

if __name__ == "__main__":
    main()