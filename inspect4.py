import foundry_local_sdk as f

f.FoundryLocalManager.initialize(f.Configuration(app_name="rag-assistant"))
manager = f.FoundryLocalManager.instance

models = manager.catalog.list_models()
for m in models:
    info = m.info
    print(f"alias={info.alias!r} id={info.id!r} size_mb={info.file_size_mb} max_output_tokens={info.max_output_tokens} device={info.runtime.device_type}")
