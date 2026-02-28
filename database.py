import sqlite3
import json
from datetime import datetime

DB_PATH = "chats.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # lets us access columns by name instead of index
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS chats (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            title       TEXT NOT NULL,
            created_at  DATETIME NOT NULL
        );

        CREATE TABLE IF NOT EXISTS messages (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id     INTEGER NOT NULL,
            role        TEXT NOT NULL,
            content     TEXT NOT NULL,
            sources     TEXT,
            created_at  DATETIME NOT NULL,
            FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
        );
    """)

    conn.commit()
    conn.close()


# --- Chat operations ---


def create_chat(title: str) -> dict:
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.utcnow().isoformat()
    cursor.execute("INSERT INTO chats (title, created_at) VALUES (?, ?)", (title, now))
    conn.commit()
    chat_id = cursor.lastrowid
    conn.close()
    return {"id": chat_id, "title": title, "created_at": now}


def get_all_chats() -> list:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM chats ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def delete_chat(chat_id: int) -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")  # needed for CASCADE to work in SQLite
    cursor.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


# --- Message operations ---


def save_message(chat_id: int, role: str, content: str, sources: list = None) -> dict:
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.utcnow().isoformat()
    sources_json = json.dumps(sources) if sources else None
    cursor.execute(
        "INSERT INTO messages (chat_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?)",
        (chat_id, role, content, sources_json, now),
    )
    conn.commit()
    msg_id = cursor.lastrowid
    conn.close()
    return {
        "id": msg_id,
        "chat_id": chat_id,
        "role": role,
        "content": content,
        "sources": sources or [],
        "created_at": now,
    }


def get_messages(chat_id: int) -> list:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM messages WHERE chat_id = ? ORDER BY created_at ASC", (chat_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    result = []
    for row in rows:
        msg = dict(row)
        msg["sources"] = json.loads(msg["sources"]) if msg["sources"] else []
        result.append(msg)
    return result
