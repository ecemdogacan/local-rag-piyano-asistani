"""Hafta 3 - Arama (retrieval): sorguya en yakin dokuman parcalarini bulma."""

from __future__ import annotations

import math

from rag.db import fetch_all_chunks, get_connection
from rag.foundry_client import FoundryClient


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (norm_a * norm_b)


def get_top_chunks(query: str, client: FoundryClient, k: int = 3) -> list[dict]:
    """
    Sorguyu embed eder, veritabanindaki tum parcalarla karsilastirir, en
    benzer k tanesini dondurur (en benzerden en az benzere siralanmis).

    Kucuk veri setleri icin butun parcalari hafizaya okuyup brute-force
    karsilastirmak yeterlidir (dokuman plani da bunu oneriyor). Veri seti
    buyudukce ozel bir vektor veritabani gerekebilir.
    """
    conn = get_connection()
    all_chunks = fetch_all_chunks(conn)
    conn.close()

    if not all_chunks:
        return []

    query_embedding = client.embed([query])[0]

    scored = [
        {**chunk, "score": cosine_similarity(query_embedding, chunk["embedding"])}
        for chunk in all_chunks
    ]
    scored.sort(key=lambda c: c["score"], reverse=True)
    return scored[:k]
