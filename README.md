# Local RAG Assistant (Piyano & Solfej)

Kullanıcının bilgisayarında tamamen **yerel (offline)** çalışan, kendi
belge koleksiyonundan soru cevaplayan bir RAG (Retrieval-Augmented
Generation) sohbet asistanı. Cevapları üretmek için Microsoft
**Foundry Local**'i (cihaz üzerinde çalışan, internet gerektirmeyen bir
yapay zeka çalıştırma ortamı) kullanır; belgeleri ve onların anlam
vektörlerini (embedding) **SQLite** ile saklar.

Bu proje, "Yaz Okulu Foundry Local Planı" müfredatının 4-6 haftalık
projesi olarak hazırlandı. Örnek/demo konu olarak **piyano çalma ve
solfej** seçildi: asistan, üç kısa belgeden (piyano temelleri, solfej
nedir, piyano pratik rutini) yararlanarak bu konudaki sorulara cevap
veriyor.

## Projenin amacı

Genel bir büyük dil modeline (ChatGPT gibi) bir soru sorduğunuzda,
model interneti/eğitim verisini kullanarak *genel* bir cevap üretir ve
bazen yanlış bilgi uydurabilir (hallüsinasyon). RAG deseni bunun yerine
şunu yapar:

1. **Retrieve (bul):** Soruya en yakın anlamdaki belge parçalarını
   kendi belge koleksiyonunuzdan bulur.
2. **Augment (zenginleştir):** Bu parçaları modele "bağlam" (context)
   olarak verir.
3. **Generate (üret):** Model, sadece bu bağlamı kullanarak cevap
   üretir; bağlamda yoksa "bilmiyorum" demesi istenir.

Sonuç: modelin cevapları sizin kendi belgelerinize dayanır, daha az
uydurma içerir ve hangi kaynaktan geldiği belirtilir.

## Proje yapısı

```
local-rag-assistant/
├── main.py                     # Calistirilacak dosya (CLI)
├── requirements.txt
├── data/
│   ├── docs/                    # Kaynak belgeler (.txt) - piyano & solfej
│   └── rag.db                    # Calisirken otomatik olusan SQLite veritabani
├── rag/
│   ├── foundry_client.py        # LLM/embedding arayuzu: FoundryClient (soyut),
│   │                             # MockFoundryClient (test icin sahte), 
│   │                             # FoundryLocalClient (gercek Foundry Local)
│   ├── db.py                     # SQLite yardimci fonksiyonlari
│   ├── ingest.py                  # chunking + embedding + kaydetme
│   ├── retrieval.py                # sorguya en yakin parcalari bulma (cosine similarity)
│   └── qa.py                        # retrieval + LLM'i birlestiren answer_query()
└── tests/
    └── test_pipeline.py            # uctan uca test (mock client ile)
```

## Mimari

```
Kullanici sorusu
      │
      ▼
[retrieval.py] sorguyu embed et, SQLite'taki parcalarla cosine similarity ile karsilastir
      │
      ▼
en alakali 2-3 parca (context)
      │
      ▼
[qa.py] system prompt + context + soruyu birlestir, Foundry Local'e gonder
      │
      ▼
cevap + kaynaklar
```

Ingestion (belgeler degistiginde veya ilk calistirmada):
```
data/docs/*.txt --> [ingest.py] paragraf paragraf chunk'lara bol --> her chunk'i embed et --> SQLite'a kaydet
```

## Mock ve gerçek mod

`rag/foundry_client.py` içindeki `FoundryClient` soyut sınıfının iki
implementasyonu var:

- **`MockFoundryClient`** — gerçek model indirmeden, deterministik sahte
  embedding/cevaplar üreterek pipeline'ı hızlıca test etmeyi sağlar.
  İnternetsiz/donanımsız her yerde çalışır.
- **`FoundryLocalClient`** — gerçek Foundry Local modellerini
  (embedding: `qwen3-embedding-0.6b`, chat: `phi-3-mini-4k`) kullanır.
  Sadece Foundry Local kurulu bir Windows/macOS bilgisayarında çalışır.

`main.py` varsayılan olarak **gerçek modeli** kullanır. Hızlı test için
mock moda geçmek isterseniz `--mock` bayrağını ekleyin. `ingest/retrieval/qa`
kodlarının hiçbiri hangi mod olduğunu bilmez — ikisi de aynı `FoundryClient`
arayüzüne karşı yazıldı.

## Kurulum

1. Foundry Local runtime'ını kurun (Windows):
   ```
   winget install Microsoft.FoundryLocal
   ```
2. Python bağımlılığını kurun:
   ```
   pip install -r requirements.txt
   ```

## Çalıştırma

```bash
cd local-rag-assistant
python main.py
```

İlk çalıştırmada modeller (ilk sefer indirilir, sonrasında önbellekten
yüklenir) ve `data/docs/` klasöründeki belgeler işlenip veritabanına
kaydedilir. Sonra terminalde soru sorabilirsiniz, örneğin:

```
Soru: Solfej nedir?
Soru: Piyano kac tusa sahiptir?
Soru: q
```

Belgeleri değiştirdiyseniz (yeni dosya ekleme/silme, içerik değişikliği)
veritabanını yeniden oluşturmak gerekir:

```bash
python main.py --reingest
```

Gerçek model olmadan hızlı test için:

```bash
python main.py --mock
```

## Testleri çalıştırma

```bash
python -m tests.test_pipeline
```

## Test sonuçları ve gözlemler (Hafta 5 - değerlendirme)

Gerçek Foundry Local ile denenen örnek sorular ve gözlemler:

| Soru | Sonuç |
|---|---|
| "Solfej nedir?" | Doğru kaynaktan (solfej_nedir.txt), akıcı ve doğru bir cevap üretti. |
| "Piyano çalarken doğru oturuş nasıl olmalı?" | Doğru kaynağı buldu (piyano_temelleri.txt) ama cevap cümleleri bazen dolambaçlı/tekrarlı çıktı. |
| Bağlamda olmayan bir soru | Sistem promptu geregi "bu konuda dokumanlarda bilgi bulamadim" demesi beklenir. |

**Gözlemlenen kısıtlar:**

- Kullanılan `phi-3-mini-4k` küçük bir modeldir (hız için tercih edildi,
  müfredatın önerdiği gibi). Basit, doğrudan bilgi sorularında ("X
  nedir?") güçlü, yorum/açıklama gerektiren sorularda ("nasıl olmalı?")
  bazen daha dağınık cevaplar veriyor. Bu, küçük/yerel modellerin bilinen
  bir sınırlaması, ChatGPT gibi büyük bulut modelleriyle kıyaslanmamalı.
- Nadiren, cevabın sonunda aynı karakterin anlamsızca tekrarlandığı
  görülüyor (örn. "0000..."). `FoundryLocalClient._clean_answer()` bunu
  büyük ölçüde temizliyor, ama nadiren küçük kalıntılar kalabiliyor.
  Kısa `max_output_tokens` (220) bu davranışı azaltıyor.
- Retrieval (doğru belgeyi bulma) adımı tutarlı ve güvenilir çalışıyor;
  asıl kısıt üretim (generation) kalitesinde.

## Bilinen sınırlamalar / geliştirme fikirleri

- Embedding'ler SQLite'ta JSON metin olarak saklanıyor (blob/özel vektör
  tipi değil) — küçük veri setleri için yeterli, büyürse özel bir vektör
  veritabanı gerekir.
- Arayüz şu an sadece komut satırı (CLI); Streamlit/Gradio ile basit bir
  web arayüzü eklenebilir (müfredatın Hafta 4 "Option B" önerisi).
- Chunk boyutu (`ingest.py` içindeki `max_chars=400`) ve getirilen parça
  sayısı (`k=3`) sabit; farklı belge türleri için ayarlanabilir.

## Müfredat planı ile eşleşme

- **Hafta 1-2:** proje iskeleti, Foundry Local kurulumu, `FoundryClient`
  arayüzü (mock + gerçek), örnek belgeler. ✅
- **Hafta 3:** ingestion (`ingest.py`) ve retrieval (`retrieval.py`,
  cosine similarity). ✅
- **Hafta 4:** gerçek Foundry Local LLM entegrasyonu (`qa.py`,
  `FoundryLocalClient`) ve CLI (`main.py`). ✅
- **Hafta 5:** test soruları, sonuçların değerlendirilmesi (yukarıdaki
  tablo), bilinen kısıtların belgelenmesi. ✅
- **Hafta 6:** bu README, kod temizliği, (isteğe bağlı) sunum ve GitHub'a
  yükleme.
