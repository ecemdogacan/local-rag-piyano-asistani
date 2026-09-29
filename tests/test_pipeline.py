"""
Uctan uca (end-to-end) test.

MockFoundryClient kullanarak: ornek dokumanlari isle -> bir soru sor ->
dogru kaynaktan cevap geldigini dogrula. Gercek bir test framework'u
(pytest) gerektirmez, sadece `python -m tests.test_pipeline` ile
calistirilir.
"""

from __future__ import annotations

from rag.db import count_chunks, get_connection, init_db
from rag.foundry_client import MockFoundryClient
from rag.ingest import ingest_documents
from rag.qa import answer_query
from rag.retrieval import cosine_similarity, get_top_chunks


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(f"BASARISIZ: {message}")
    print(f"OK: {message}")


def test_cosine_similarity_basics() -> None:
    check(cosine_similarity([1, 0], [1, 0]) > 0.99, "ayni vektorun benzerligi ~1 olmali")
    check(abs(cosine_similarity([1, 0], [0, 1])) < 0.01, "dik vektorlerin benzerligi ~0 olmali")


def test_ingest_and_retrieve() -> None:
    client = MockFoundryClient()

    total = ingest_documents(client)
    check(total > 0, f"ingest_documents en az bir parca kaydetmeli (kaydedilen: {total})")

    conn = get_connection()
    init_db(conn)
    count = count_chunks(conn)
    conn.close()
    check(count == total, "veritabanindaki parca sayisi ingest sonucuyla eslesmeli")

    top = get_top_chunks("Piyano calarken dogru oturus nasil olmali?", client, k=2)
    check(len(top) > 0, "en az bir sonuc donmeli")
    check(
        any("piyano_temelleri" in c["source"] for c in top),
        f"'Piyano calarken dogru oturus nasil olmali?' sorusu piyano_temelleri.txt dosyasini bulmali, bulunanlar: {[c['source'] for c in top]}",
    )


def test_answer_query_end_to_end() -> None:
    client = MockFoundryClient()
    ingest_documents(client)

    result = answer_query("Solfej nedir?", client, k=2)
    check(bool(result["answer"]), "bir cevap donmeli")
    check(len(result["sources"]) > 0, "en az bir kaynak donmeli")
    check(
        any("solfej" in s for s in result["sources"]),
        f"cevap solfej_nedir.txt kaynagini icermeli, bulunanlar: {result['sources']}",
    )

    empty_client_result = answer_query("Bu soru dokumanlarla hic alakasiz olsun mu?", client, k=2)
    check(bool(empty_client_result["answer"]), "alakasiz sorularda bile bir cevap donmeli (mock icin)")


def main() -> None:
    test_cosine_similarity_basics()
    test_ingest_and_retrieve()
    test_answer_query_end_to_end()
    print("\nTum testler basarili.")


if __name__ == "__main__":
    main()
