#!/usr/bin/env python3
import os
import time
import logging
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4o-mini"  # Language model for generating responses
PERSIST_DIRECTORY = "./faiss_index"
TARGET_SOURCE_CHUNKS = 4

load_dotenv()

def create_financial_advice_chain():
    """
    Create a chain that retrieves documents and generates financial advice.
    """
    try:
        # Initialize embeddings and vector store
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        logger.info(f"Loading FAISS index from {PERSIST_DIRECTORY}")
        vectorstore = FAISS.load_local(PERSIST_DIRECTORY, embeddings, allow_dangerous_deserialization=True)
        
        # Initialize language model
        logger.info(f"Initializing language model: {LLM_MODEL}")
        llm = ChatOpenAI(model_name=LLM_MODEL, temperature=0.7, timeout=120)  # Increased timeout
        
        # Define a prompt template for financial advice
        prompt_template = PromptTemplate.from_template(
            """You are the AI representative of a knowledgeable content creator, speaking in their authentic voice and drawing from their published content.

            Content Creator's Published Material:
            {context}

            User Question: {question}

            Instructions:
            - Maintain the original creator's tone, style, and perspective
            - Reference specific articles or posts when relevant
            - If the context doesn't contain relevant information, draw from the general themes and approach present in the creator's content
            - Be transparent about when you're making general statements vs directly quoting content
            - Maintain consistency with the creator's known positions and values
            - Include relevant examples or anecdotes from the creator's content when applicable
    
            Response Format:
            1. Direct Answer
            2. Supporting Context (with specific references when available)
            3. Related Content Recommendations (if applicable)

            Response:"""
        )
        
        # Create the retrieval chain
        retriever = vectorstore.as_retriever(search_kwargs={"k": TARGET_SOURCE_CHUNKS})
        
        def format_docs(docs):
            if not docs:
                return "No relevant documents found."
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
        query = input("Enter your financial query: ")
        
        if query.lower() == "exit":
            print("\nExiting chatbot. Goodbye!")
            break
        
        if not query.strip():
            print("Empty query. Please ask something meaningful.")
            continue

        print("\nAnalyzing documents and generating advice...")
        
        try:
            # Generate financial advice with a timeout mechanism
            start_time = time.time()
            advice = advice_chain.invoke(query)
            
            print("\n--- Financial Advice ---")
            print(advice)
            print("\n--- End of Advice ---")
            
            logger.info(f"Query processed in {time.time() - start_time:.2f} seconds")
        
        except Exception as e:
            print(f"Error generating advice: {e}")
            logger.error(f"Advice generation error: {e}")

if __name__ == "__main__":
    main()