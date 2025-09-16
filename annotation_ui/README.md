# Annotation UI (Streamlit)

A tiny local UI to draw boxes and set attributes (permanence, zone, permit_status, area_type, structure_type) per object, then export to JSONL compatible with tools/ontology_mapper.py.

Run:
```
streamlit run annotation_ui/app.py
```

Output:
- Writes a JSONL file you can feed to `tools/ontology_mapper.py`.
