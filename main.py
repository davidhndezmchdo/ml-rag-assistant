import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

from database import (
    init_db,
    create_chat,
    get_all_chats,
    delete_chat,
    save_message,
    get_messages,
)

# Configure models
Settings.llm = Ollama(
    model="llama3.1:8b",
    request_timeout=120.0,
    context_window=2048,
    system_prompt="You are a helpful research assistant. Answer questions based on the provided context from ML research papers. Be concise and precise. Do not include meta-instructions or rewriting notes in your response.",
)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# Load index from disk
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("ml_papers")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(
    vector_store, storage_context=storage_context
)
query_engine = index.as_query_engine(similarity_top_k=3)

# Init database on startup
init_db()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic models ---


class NewChatRequest(BaseModel):
    title: Optional[str] = "New Chat"


class QueryRequest(BaseModel):
    question: str


# --- Chat endpoints ---


@app.get("/chats")
def list_chats():
    return get_all_chats()


@app.post("/chats")
def new_chat(request: NewChatRequest):
    return create_chat(request.title)


@app.delete("/chats/{chat_id}")
def remove_chat(chat_id: int):
    deleted = delete_chat(chat_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"deleted": True}


@app.get("/chats/{chat_id}/messages")
def load_messages(chat_id: int):
    return get_messages(chat_id)


@app.post("/chats/{chat_id}/query")
def query(chat_id: int, request: QueryRequest):
    # Verify chat exists
    chats = get_all_chats()
    if not any(c["id"] == chat_id for c in chats):
        raise HTTPException(status_code=404, detail="Chat not found")

    # Save user message
    save_message(chat_id, "user", request.question)

    # Run RAG
    response = query_engine.query(request.question)

    # Extract sources
    sources = []
    for node in response.source_nodes:
        filename = node.metadata.get("file_name", "Unknown")
        if filename not in sources:
            sources.append(filename)

    # Save assistant message
    answer = (
            str(response).replace("**Rewrite**", "").replace("**Rewrite:**", "").replace("Rewrite", "").replace("Rewrite:", "").strip()
    )
    save_message(chat_id, "assistant", answer, sources)

    # Auto-title the chat after first message
    messages = get_messages(chat_id)
    if len(messages) == 2:  # first user + first assistant = 2
        short_title = request.question[:50] + (
            "..." if len(request.question) > 50 else ""
        )
        from database import get_connection

        conn = get_connection()
        conn.execute("UPDATE chats SET title = ? WHERE id = ?", (short_title, chat_id))
        conn.commit()
        conn.close()

    return {"answer": answer, "sources": sources, "title": short_title if len(messages) == 2 else None}


@app.get("/health")
def health():
    return {"status": "ok"}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
