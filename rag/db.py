"""SQLite yardimci fonksiyonlari: dokuman parcalarini (chunk) ve embedding'lerini
saklama / okuma.

Embedding vektorlerini basitlik icin JSON metin olarak saklıyoruz (SQLite'in
yerlesik bir vektor tipi yok). Kucuk veri setleri icin bu tamamen yeterlidir.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "rag.db"


def get_connection(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            chunk_text TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
        """
    )
    conn.commit()


def clear_chunks(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM chunks")
    conn.commit()


def insert_chunk(conn: sqlite3.Connection, source: str, chunk_text: str, embedding: list[float]) -> None:
    conn.execute(
        "INSERT INTO chunks (source, chunk_text, embedding) VALUES (?, ?, ?)",
        (source, chunk_text, json.dumps(embedding)),
    )


def fetch_all_chunks(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT id, source, chunk_text, embedding FROM chunks").fetchall()
    return [
        {"id": r[0], "source": r[1], "chunk_text": r[2], "embedding": json.loads(r[3])}
        for r in rows
    ]


def count_chunks(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]


def peek_embedding_dim(conn: sqlite3.Connection) -> int | None:
    """Veritabanindaki ilk parcanin embedding uzunlugunu dondurur (yoksa None).

    Mock ve gercek modelin embedding boyutlari farkli oldugu icin, hangi
    modelle doldurulmus bir veritabaniyla karsi karsiya oldugumuzu anlamak
    icin kullanilir.
    """
    row = conn.execute("SELECT embedding FROM chunks LIMIT 1").fetchone()
    if row is None:
        return None
    return len(json.loads(row[0]))
