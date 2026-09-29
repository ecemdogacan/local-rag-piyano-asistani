import inspect
import foundry_local_sdk as f

print("Catalog.get_model sig:", inspect.signature(f.Catalog.get_model))
print("Catalog.get_model doc:", inspect.getdoc(f.Catalog.get_model))
print()
print("Parameter sig:", inspect.signature(f.Parameter))
print("Parameter doc:", inspect.getdoc(f.Parameter))
print()
f.FoundryLocalManager.initialize(f.Configuration(app_name="rag-assistant"))
manager = f.FoundryLocalManager.instance
model = manager.catalog.get_model("phi-3.5-mini")
print("context_length:", model.context_length)
print("info:", model.info)
print("capabilities:", model.capabilities)
