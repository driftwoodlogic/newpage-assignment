# Processed Dataset

Files:
- `people_2k.ndjson`: 2,000 sampled people records.
- `chunks_2k.jsonl`: chunked passages for RAG retrieval.

Chunk schema (JSONL):
- `chunk_id`: stable id, format `source_id:type:index`
- `source_id`: original record identifier
- `name`: person name
- `url`: Wikipedia URL
- `section`: section or infobox field name
- `type`: `abstract` | `infobox` | `section`
- `text`: chunk text
- Optional: `field_name`, `fact_value` (infobox only)
