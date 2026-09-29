#!/usr/bin/env python3
"""
Local RAG Assistant - giris noktasi.

Kullanim:
    python main.py              gercek Foundry Local modelleriyle calisir
                                 (dokumanlari isler, gerekirse) ve soru-cevap
                                 dongusune girer
    python main.py --reingest   veritabanini sifirlayip dokumanlari
                                 yeniden isler
    python main.py --mock       gercek model yerine sahte (mock) cevaplarla
                                 hizli test yapmak icin
"""

from __future__ import annotations

import sys

from rag.db import count_chunks, get_connection, init_db, peek_embedding_dim
from rag.ingest import ingest_documents
from rag.qa import answer_query


def ensure_ingested(client, force: bool = False) -> None:
    conn = get_connection()
    init_db(conn)
    existing = count_chunks(conn)
    existing_dim = peek_embedding_dim(conn)
    conn.close()

    if existing > 0 and not force:
        current_dim = len(client.embed(["test"])[0])
        if existing_dim is not None and existing_dim != current_dim:
            print(
                f"[bilgi] Veritabanindaki embedding boyutu ({existing_dim}) "
                f"su anki modelinkiyle ({current_dim}) uyusmuyor -- muhtemelen "
                f"mock ile gercek model karisti. Otomatik olarak yeniden "
                f"isleniyor.\n"
            )
            force = True
        else:
            print(
                f"[bilgi] Veritabaninda zaten {existing} parca var, ingest atlaniyor "
                f"(yeniden islemek icin: python main.py --reingest)"
            )
            return

    print("[bilgi] Dokumanlar isleniyor (chunk + embed + kaydet)...")
    total = ingest_documents(client)
    print(f"[bilgi] Tamamlandi: {total} parca veritabanina kaydedildi.\n")


def run_cli(client) -> None:
    print("=== Local RAG Assistant ===")
    print("Cikmak icin 'q' yazip Enter'a basin.\n")
    while True:
        try:
            question = input("Soru: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGorusmek uzere!")
            break

        if question.lower() in {"q", "quit", "exit"}:
            print("Gorusmek uzere!")
            break
        if not question:
            continue

        result = answer_query(question, client)
        print(f"\nCevap: {result['answer']}")
        if result["sources"]:
            print(f"Kaynaklar: {', '.join(result['sources'])}")
        print()


def main() -> None:
    force = "--reingest" in sys.argv
    use_mock = "--mock" in sys.argv

    if use_mock:
        from rag.foundry_client import MockFoundryClient

        print("[bilgi] Mock (sahte) mod: gercek model kullanilmiyor.\n")
        client = MockFoundryClient()
        # Not: mock ve gercek modelin embedding boyutlari farkli oldugu icin,
        # ensure_ingested bu karisikligi kendisi tespit edip otomatik olarak
        # yeniden isler -- burada elle force=True yapmaya gerek yok.
    else:
        from rag.foundry_client import FoundryLocalClient

        print("[bilgi] Gercek Foundry Local modelleri baslatiliyor "
              "(ilk calistirmada model indirme suresi alabilir)...\n")
        client = FoundryLocalClient()

    ensure_ingested(client, force=force)
    run_cli(client)


if __name__ == "__main__":
    main()
