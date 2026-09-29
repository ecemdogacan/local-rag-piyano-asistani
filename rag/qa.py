"""Hafta 4 - Retrieval + LLM entegrasyonu: answer_query()"""

from __future__ import annotations

from rag.foundry_client import FoundryClient
from rag.retrieval import get_top_chunks

SYSTEM_PROMPT = (
    "Sen, verilen baglam (context) disina cikmayan bir soru-cevap "
    "asistanisin. Sadece BAGLAM icinde verilen bilgiyi kullanarak cevap "
    "ver. Eger cevap baglamda yoksa, 'Bu konuda dokumanlarda bilgi "
    "bulamadim' de ve bilgi uydurma. Cevabinin sonunda hangi kaynaktan "
    "(dosya adindan) yararlandigini belirt."
)


def answer_query(question: str, client: FoundryClient, k: int = 3) -> dict:
    """
    Soruyu cevaplar.

    Donen sozluk:
      - "answer": modelin cevabi (str)
      - "sources": kullanilan kaynak dosya adlari (list[str])
      - "chunks": kullanilan ham metin parcalari, debug/log icin (list[dict])
    """
    top_chunks = get_top_chunks(question, client, k=k)

    if not top_chunks:
        return {
            "answer": "Veritabaninda hic dokuman yok. Once ingest_documents() calistirin.",
            "sources": [],
            "chunks": [],
        }

    context = "\n".join(f"- ({c['source']}) {c['chunk_text']}" for c in top_chunks)
    user_prompt = f"BAGLAM:\n{context}\n\nSORU:\n{question}"

    answer = client.chat(SYSTEM_PROMPT, user_prompt)
    sources = sorted({c["source"] for c in top_chunks})

    return {"answer": answer, "sources": sources, "chunks": top_chunks}
