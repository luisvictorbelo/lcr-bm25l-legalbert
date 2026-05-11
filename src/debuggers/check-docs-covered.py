import json
from pathlib import Path


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

cwd = Path.cwd()

query_path = cwd / 'data' / 'processed' / 'paragraphs' / 'query_paragraphs_all.json'

corpus_path = cwd / 'data' / 'processed' / 'paragraphs' / 'corpus_paragraphs_all.json'


query = load(query_path)
corpus = load(corpus_path)

query_ids = {x["doc_id"] for x in query}
corpus_ids = {x["doc_id"] for x in corpus}

overlap = query_ids & corpus_ids
union = query_ids | corpus_ids

print("query docs:", len(query_ids))
print("corpus docs:", len(corpus_ids))
print("overlap:", len(overlap))
print("unique total:", len(union))