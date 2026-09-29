"""
Foundry Local ile konusan katman.

Bu dosyada iki sey var:
1. FoundryClient: LLM ve embedding cagrilarinin ne yapmasi gerektigini
   tanimlayan bir "sozlesme" (interface / soyut sinif).
2. MockFoundryClient: gercek Foundry Local kurulu olmadan (ornegin bu
   bulut ortaminda) pipeline'in geri kalanini test edebilmek icin sahte
   ama tutarli cevaplar ureten bir implementasyon.

Asagida ayrica FoundryLocalClient adinda, gercek Foundry Local'e
baglanacaginiz zaman doldurmaniz gereken bir iskelet var. Onu
tamamladiginizda main.py icinde tek satir degistirerek (MockFoundryClient
-> FoundryLocalClient) gercek modele gecebilirsiniz; ingest/retrieval/qa
kodlarinin hicbiri degismez.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod


class FoundryClient(ABC):
    """LLM ve embedding saglayicilarinin uymasi gereken arayuz (interface)."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Her metin icin sabit uzunlukta sayisal bir vektor (embedding) dondurur."""
        raise NotImplementedError

    @abstractmethod
    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """System + user prompt verilince modelin metin cevabini dondurur."""
        raise NotImplementedError


class MockFoundryClient(FoundryClient):
    """
    Gercek bir model indirmeden calisan sahte (mock) implementasyon.

    - embed(): kelime frekanslarina dayali basit, deterministik bir vektor
      uretir. Gercek bir embedding modeli kadar "anlami" yakalamaz, ama
      ayni kelimeleri paylasan metinleri birbirine yaklastirir -- bu da
      pipeline'i (ingest -> retrieval -> qa) uctan uca test etmek icin
      yeterlidir.
    - chat(): gercek bir LLM gibi serbestce konusmaz; bunun yerine
      kendisine verilen baglami (context) kullanarak kisa, ongorulebilir
      bir cevap uretir. Boylece "dogru baglam modele ulasti mi" sorusunu
      test edebiliriz.
    """

    VOCAB_SIZE = 256

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.VOCAB_SIZE
        words = re.findall(r"[a-zA-ZçÇğĞıİöÖşŞüÜ]+", text.lower())
        for word in words:
            idx = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % self.VOCAB_SIZE
            vector[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        context_match = re.search(r"BAGLAM:\n(.*?)\n\nSORU:", user_prompt, re.S)
        question_match = re.search(r"SORU:\n(.*)", user_prompt, re.S)
        context = context_match.group(1).strip() if context_match else ""
        question = question_match.group(1).strip() if question_match else user_prompt

        if not context:
            return "Bu konuda dokumanlarda bilgi bulamadim, bu yuzden emin degilim."

        snippet = context.split("\n")[0][:220]
        return f"[MOCK CEVAP] '{question}' sorusuna dokumanlarda bulunan bilgiye gore: {snippet}..."


class FoundryLocalClient(FoundryClient):
    """
    Gercek Foundry Local baglantisi -- SADECE kendi bilgisayarinizda
    (Windows/macOS) calisir, bu bulut ortaminda calismaz.

    Kurulum (kendi bilgisayarinizda):
        winget install Microsoft.FoundryLocal
        pip install foundry-local-sdk

    Not: chat_model varsayilan olarak "phi-3-mini-4k" -- "phi-3.5-mini"
    DEGIL. phi-3.5-mini'nin varsayilan baglam penceresi (128k token) cok
    buyuk oldugu icin bazi bilgisayarlarda bellek hatasi (KV-cache
    allocation failed) veriyor. phi-3-mini-4k ayni aile icinde daha kucuk
    (4k token) bir surum ve bu sorunu yasamiyor.
    """

    def __init__(
        self,
        chat_model: str = "phi-3-mini-4k",
        embedding_model: str = "qwen3-embedding-0.6b",
    ) -> None:
        import foundry_local_sdk as foundry

        self._foundry = foundry
        self.chat_model = chat_model
        self.embedding_model = embedding_model

        foundry.FoundryLocalManager.initialize(
            foundry.Configuration(app_name="rag-assistant")
        )
        self.manager = foundry.FoundryLocalManager.instance

        self._embed_model = self._get_ready_model(embedding_model)
        self._chat_model = self._get_ready_model(chat_model)

    def _get_ready_model(self, alias: str):
        """Modeli katalogdan bulur, gerekirse indirir, gerekirse yukler."""
        model = self.manager.catalog.get_model(alias)
        if not model.is_cached:
            print(f"'{alias}' modeli ilk kez kullaniliyor, indiriliyor...")
            model.download()
        model.load()
        return model

    def embed(self, texts: list[str]) -> list[list[float]]:
        import array

        f = self._foundry
        vectors: list[list[float]] = []

        with f.EmbeddingsSession(self._embed_model) as session:
            for text in texts:
                with f.Request().add_item(f.TextItem(text)) as req:
                    with session.process_request(req) as resp:
                        for item in resp:
                            # item bir TensorItem: .data ham float32 byte'lari.
                            floats = array.array("f", item.data).tolist()
                            vectors.append(floats)

        return vectors

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        f = self._foundry

        with f.ChatSession(self._chat_model) as session:
            session.set_options(
                f.RequestOptions(
                    # Kucuk modeller bazen cevabi bitirdikten sonra ayni
                    # karakteri/kelimeyi tekrar tekrar uretmeye devam
                    # edebiliyor ("bozulma"/degenerate cikti). Izin verilen
                    # token sayisini dusuk tutmak bunu byuk olcude azaltiyor;
                    # kalanini asagida _clean_answer temizliyor.
                    search=f.SearchOptions(temperature=0.0, max_output_tokens=220),
                )
            )
            request = f.Request()
            if hasattr(f.MessageItem, "system"):
                request.add_item(f.MessageItem.system(system_prompt))
                request.add_item(f.MessageItem.user(user_prompt))
            else:
                # Bu SDK surumunde ayri bir "system" mesaj tipi yoksa,
                # sistem talimatini kullanici mesajinin basina ekliyoruz.
                combined = f"{system_prompt}\n\n{user_prompt}"
                request.add_item(f.MessageItem.user(combined))
            with request as req:
                with session.process_request(req) as resp:
                    answer_parts = []
                    for item in resp:
                        if isinstance(item, f.MessageItem):
                            answer_parts.append(item.get_simple_text())

        return self._clean_answer("".join(answer_parts))

    @staticmethod
    def _clean_answer(text: str) -> str:
        """Modelin bazen cevabin sonuna ekledigi anlamsiz tekrarlari keser.

        Ornek: ayni karakterin 3+ kez ust uste tekrarlandigi noktadan
        itibaren metni kesiyoruz (ornegin "0000" gibi) -- normal bir
        Turkce cumlede boyle bir tekrar gecmez, bu yuzden bunu her zaman
        modelin "bozulmaya" basladigi yer olarak kabul edebiliriz. Rakam
        disi karakterlerde (harf/noktalama) de ayni kural gecerli.
        """
        match = re.search(r"(.)\1{2,}", text)
        if match:
            text = text[: match.start()]
        return text.strip()
