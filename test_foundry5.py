import foundry_local_sdk as f

print("Foundry Local baslatiliyor...")
f.FoundryLocalManager.initialize(f.Configuration(app_name="rag-assistant"))
manager = f.FoundryLocalManager.instance

print("Embedding modeli aliniyor: qwen3-embedding-0.6b")
embed_model = manager.catalog.get_model("qwen3-embedding-0.6b")
print("cached:", embed_model.is_cached)
if not embed_model.is_cached:
    print("Indiriliyor (internet gerekiyor, birkac dakika surebilir)...")
    embed_model.download()
embed_model.load()
print("Embedding modeli yuklendi.")

with f.EmbeddingsSession(embed_model) as session:
    with f.Request().add_item(f.TextItem("merhaba dunya")) as req:
        with session.process_request(req) as resp:
            for item in resp:
                print("ITEM TYPE:", type(item).__name__)
                print("ITEM ATTRS:", [a for a in dir(item) if not a.startswith("_")])
                if hasattr(item, "__dict__"):
                    print("ITEM DICT:", item.__dict__)

print()
print("Chat modeli aliniyor: phi-3-mini-4k")
chat_model = manager.catalog.get_model("phi-3-mini-4k")
print("cached:", chat_model.is_cached)
if not chat_model.is_cached:
    print("Indiriliyor (bu daha buyuk olabilir, biraz bekleyin)...")
    chat_model.download()
chat_model.load()
print("Chat modeli yuklendi.")

with f.ChatSession(chat_model) as session:
    session.set_options(f.RequestOptions(
        search=f.SearchOptions(temperature=0.0, max_output_tokens=50),
        additional_options={"max_length": "2048"},
    ))
    with f.Request().add_item(f.MessageItem.user("7 carpi 6 kactir?")) as req:
        with session.process_request(req) as resp:
            for item in resp:
                print("RESP ITEM TYPE:", type(item).__name__)
                print("RESP ITEM ATTRS:", [a for a in dir(item) if not a.startswith("_")])
                if isinstance(item, f.TextItem):
                    print("CEVAP (TextItem):", item.text)
                if isinstance(item, f.MessageItem):
                    print("is_simple_text:", item.is_simple_text)
                    print("get_simple_text:", item.get_simple_text())
                    print("parts:", item.parts)

print("TAMAM.")
