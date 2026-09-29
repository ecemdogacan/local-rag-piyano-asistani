"""
Hafta 3 - Veri hazirlama (ingestion) hatti.

Adimlar:
1. data/docs/ klasorundeki her .txt dosyasini oku.
2. Metni kucuk parcalara (chunk) bol.
3. Her parca icin embedding hesapla (FoundryClient uzerinden).
4. Parcayi ve embedding'ini SQLite'a kaydet.
"""

from __future__ import annotations

from pathlib import Path

from rag.db import clear_chunks, get_connection, init_db, insert_chunk
from rag.foundry_client import FoundryClient

DEFAULT_DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"


def chunk_text(text: str, max_chars: int = 400) -> list[str]:
    """
    Metni once bos satirlara gore paragraflara boler, sonra max_chars'tan
    uzun paragraflari da esit parcalara ayirir.

    Not: Bu, egitim amacli basit bir yontemdir. Gercek projelerde cumle
    sinirlarina veya baslik yapisina gore daha akilli bolme yapilabilir.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    for para in paragraphs:
        if len(para) <= max_chars:
            chunks.append(para)
        else:
            for i in range(0, len(para), max_chars):
                chunks.append(para[i : i + max_chars])
    return chunks


def ingest_documents(client: FoundryClient, docs_dir: Path = DEFAULT_DOCS_DIR) -> int:
    """docs_dir icindeki tum .txt dosyalarini isleyip veritabanina kaydeder.

    Donen deger: kaydedilen toplam chunk sayisi.
    """
    conn = get_connection()
    init_db(conn)
    clear_chunks(conn)

    total = 0
    for file_path in sorted(docs_dir.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        if not chunks:
            continue
        embeddings = client.embed(chunks)
        for chunk, embedding in zip(chunks, embeddings):
            insert_chunk(conn, source=file_path.name, chunk_text=chunk, embedding=embedding)
            total += 1

    conn.commit()
    conn.close()
    return total
