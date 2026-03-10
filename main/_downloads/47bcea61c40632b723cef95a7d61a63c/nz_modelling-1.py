from binny import NZTomography

models = NZTomography.list_nz_models()

print(f"Found {len(models)} registered n(z) models:")
for name in models:
    print(f" - {name}")