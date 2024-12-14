from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore




raw_documents = TextLoader('hasolidit_articles.json').load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

embeddings = OllamaEmbeddings(model="llama3")


vector_store = FAISS(
    embedding_function=embeddings,
    index="faiss_index",
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

vector_store.save_local("hasolidit")
new_vector_store = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)

db = FAISS.from_documents(documents, OllamaEmbeddings())

