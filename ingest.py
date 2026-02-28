import chromadb
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

# Configure models
Settings.llm = Ollama(model="llama3.1:8b", request_timeout=120.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# Load docs
print("Loading documents......")
documents = SimpleDirectoryReader("docs").load_data()
print(f"Loaded {len(documents)} document chunks")

# Set up ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("ml_papers")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)


# Build index
print("Building index... (this will take a few minutes)")
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, show_progress=True
)

print("Done! Index saved to ./chroma_db")
