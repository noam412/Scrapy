from langchain_ollama.llms import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings


# Local imports
from .models.prompts import init_prompt

template = """Question: {question}

Answer: Let's think step by step.
"""

prompt = init_prompt()
model = OllamaLLM(base_url="https://10.100.102.10:11434", model="llama3.1")

chain = prompt | model

chain.invoke({"question": "What is LangChain?"})

