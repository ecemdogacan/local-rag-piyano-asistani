import inspect
import foundry_local_sdk as f

print("Model.load sig:", inspect.signature(f.Model.load))
print("Model.load doc:", inspect.getdoc(f.Model.load))
print()
print("Model.select_variant sig:", inspect.signature(f.Model.select_variant))
print("Model.select_variant doc:", inspect.getdoc(f.Model.select_variant))
print()
print("ModelSettings sig:", inspect.signature(f.ModelSettings))
print("ModelSettings doc:", inspect.getdoc(f.ModelSettings))
