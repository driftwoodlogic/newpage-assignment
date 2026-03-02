#!/usr/bin/env python3
"""Build a 2k-people RAG dataset and a 100-question eval set.

Inputs:
  - data/people_*.ndjson (English Wikipedia People dataset)
Outputs:
  - data/processed/people_2k.ndjson
  - data/processed/chunks_2k.jsonl
  - eval/questions_100.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
VERB_RE = re.compile(
    r"\b(was|were|began|joined|released|won|signed|appointed|retired|married|served|published|graduated|debuted|earned|elected|promoted|returned|represented|completed|founded|named|selected)\b",
    re.IGNORECASE,
)


def stable_hash(value: str, seed: str) -> int:
    h = hashlib.blake2b(f"{seed}:{value}".encode("utf-8"), digest_size=8)
    return int.from_bytes(h.digest(), "big")


def read_ndjson(paths: List[Path]) -> Iterable[Dict[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def select_sample(
    records: Iterable[Dict[str, Any]], sample_size: int, seed: str
) -> List[Dict[str, Any]]:
    # Maintain a max-heap of size sample_size using hash score.
    heap: List[Tuple[int, Dict[str, Any]]] = []
    for rec in records:
        identifier = rec.get("identifier")
        if identifier is None:
            # Fall back to name for stability if missing.
            identifier = rec.get("name", "")
        h = stable_hash(str(identifier), seed)
        if len(heap) < sample_size:
            heapq.heappush(heap, (-h, rec))
        else:
            if h < -heap[0][0]:
                heapq.heapreplace(heap, (-h, rec))
    selected = [rec for _, rec in heap]
    selected.sort(key=lambda r: stable_hash(str(r.get("identifier", r.get("name", ""))), seed))
    return selected


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def truncate(text: str, max_len: int = 1200) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def iter_infobox_fields(record: Dict[str, Any]) -> Iterable[Tuple[str, str]]:
    for infobox in record.get("infobox", []) or []:
        for section in infobox.get("has_parts", []) or []:
            for field in section.get("has_parts", []) or []:
                if field.get("type") != "field":
                    continue
                name = field.get("name") or "Field"
                value = field.get("value")
                if not value:
                    values = field.get("values") or []
                    value = " ".join(v for v in values if v)
                if not value:
                    continue
                yield name, value


def iter_section_paragraphs(record: Dict[str, Any]) -> Iterable[Tuple[str, str]]:
    for section in record.get("article_sections", []) or []:
        section_name = section.get("name") or "Section"
        if section_name.lower() == "abstract":
            continue
        for part in section.get("has_parts", []) or []:
            ptype = part.get("type")
            if ptype == "paragraph":
                value = part.get("value") or ""
                if value:
                    yield section_name, value
            elif ptype == "list":
                items = []
                for item in part.get("has_parts", []) or []:
                    if item.get("type") == "list_item":
                        iv = item.get("value") or ""
                        if iv:
                            items.append(iv)
                if items:
                    yield section_name, " ".join(items)


def build_chunks(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for rec in records:
        name = rec.get("name", "")
        url = rec.get("url", "")
        identifier = rec.get("identifier")
        if identifier is None:
            identifier = name
        source_id = str(identifier)

        abstract = rec.get("abstract") or ""
        if abstract:
            text = truncate(clean_text(abstract))
            if len(text) >= 40:
                chunks.append(
                    {
                        "chunk_id": f"{source_id}:abstract",
                        "source_id": source_id,
                        "name": name,
                        "url": url,
                        "section": "Abstract",
                        "type": "abstract",
                        "text": text,
                    }
                )

        infobox_idx = 0
        for field_name, value in iter_infobox_fields(rec):
            text = truncate(clean_text(f"{field_name}: {value}"))
            if len(text) < 20:
                continue
            chunks.append(
                {
                    "chunk_id": f"{source_id}:infobox:{infobox_idx}",
                    "source_id": source_id,
                    "name": name,
                    "url": url,
                    "section": f"Infobox:{field_name}",
                    "type": "infobox",
                    "field_name": field_name,
                    "fact_value": value,
                    "text": text,
                }
            )
            infobox_idx += 1

        section_idx = 0
        for section_name, paragraph in iter_section_paragraphs(rec):
            text = truncate(clean_text(paragraph))
            if len(text) < 40:
                continue
            chunks.append(
                {
                    "chunk_id": f"{source_id}:section:{section_idx}",
                    "source_id": source_id,
                    "name": name,
                    "url": url,
                    "section": section_name,
                    "type": "section",
                    "text": text,
                }
            )
            section_idx += 1

    return chunks


def format_infobox_question(name: str, field_name: str) -> str:
    f = field_name.strip().lower()
    if f in {"born", "birth"}:
        return f"When and where was {name} born?"
    if f in {"date of birth"}:
        return f"When was {name} born?"
    if f in {"place of birth", "birth place"}:
        return f"Where was {name} born?"
    if f in {"died", "date of death"}:
        return f"When did {name} die?"
    if f in {"place of death"}:
        return f"Where did {name} die?"
    if "occupation" in f:
        return f"What is {name}'s occupation?"
    if f in {"nationality"}:
        return f"What is {name}'s nationality?"
    if f in {"education", "alma mater"}:
        return f"Where did {name} study?"
    if f in {"spouse", "partner"}:
        return f"Who is {name}'s spouse or partner?"
    if f in {"children"}:
        return f"How many children does {name} have?"
    if f in {"current team", "team", "teams"}:
        return f"Which team does {name} play for?"
    if f in {"position", "position(s)"}:
        return f"What position does {name} play?"
    if f in {"known for"}:
        return f"What is {name} known for?"
    if f in {"venerated in"}:
        return f"Where is {name} venerated?"
    return f"What is {name}'s {field_name}?"


def find_year_sentence(text: str) -> Optional[Tuple[str, str]]:
    # Return (year, sentence) if a sentence contains a year and a likely verb.
    for sentence in SENT_SPLIT_RE.split(text):
        m = YEAR_RE.search(sentence)
        if not m:
            continue
        if not VERB_RE.search(sentence):
            continue
        s = sentence.strip()
        if len(s.split()) < 8:
            continue
        year = m.group(0)
        return year, s
    return None


def name_tokens(name: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9'-]+", name)
    return [t for t in tokens if len(t) >= 3]


def sentence_mentions_subject(sentence: str, name: str) -> bool:
    tokens = name_tokens(name)
    if tokens and any(re.search(rf"\\b{re.escape(t)}\\b", sentence, re.IGNORECASE) for t in tokens):
        return True
    # Allow pronoun-led sentences common in biographies.
    if re.match(r"^(He|She|They|His|Her|Their|The)\b", sentence.strip()):
        return True
    # Allow time-led sentences: "In 2001, ..." / "On 12 May 2010, ..."
    if re.match(r"^(In|On|By|During)\s+\d{4}", sentence.strip()):
        return True
    return False


def build_eval_set(
    chunks: List[Dict[str, Any]],
    total_questions: int,
    seed: int,
    infobox_ratio: float = 0.6,
) -> List[Dict[str, Any]]:
    rnd = random.Random(seed)

    infobox_chunks = [c for c in chunks if c.get("type") == "infobox" and c.get("fact_value")]
    narrative_chunks = [c for c in chunks if c.get("type") == "section"]

    infobox_target = int(total_questions * infobox_ratio)
    narrative_target = total_questions - infobox_target

    rnd.shuffle(infobox_chunks)
    rnd.shuffle(narrative_chunks)

    questions: List[Dict[str, Any]] = []

    # Infobox questions
    for c in infobox_chunks:
        if len(questions) >= infobox_target:
            break
        name = c.get("name", "")
        field_name = c.get("field_name", "")
        value = c.get("fact_value", "")
        if not name or not field_name or not value:
            continue
        q = format_infobox_question(name, field_name)
        questions.append(
            {
                "id": f"q{len(questions)+1:03d}",
                "question": q,
                "gold_answer": clean_text(value),
                "gold_chunk_ids": [c.get("chunk_id")],
                "gold_context": c.get("text"),
                "source_id": c.get("source_id"),
                "source_url": c.get("url"),
                "type": "infobox",
            }
        )

    # Narrative questions
    for c in narrative_chunks:
        if len(questions) >= total_questions:
            break
        name = c.get("name", "")
        if not name:
            continue
        year_sentence = find_year_sentence(c.get("text", ""))
        if not year_sentence:
            continue
        year, sentence = year_sentence
        if not sentence_mentions_subject(sentence, name):
            continue
        q = f"What happened to {name} in {year}?"
        questions.append(
            {
                "id": f"q{len(questions)+1:03d}",
                "question": q,
                "gold_answer": truncate(clean_text(sentence), max_len=300),
                "gold_chunk_ids": [c.get("chunk_id")],
                "gold_context": c.get("text"),
                "source_id": c.get("source_id"),
                "source_url": c.get("url"),
                "type": "narrative",
            }
        )

    # If not enough narrative questions, top up with infobox ones.
    if len(questions) < total_questions:
        needed = total_questions - len(questions)
        remaining = [c for c in infobox_chunks if c.get("chunk_id") not in {q["gold_chunk_ids"][0] for q in questions}]
        for c in remaining[:needed]:
            name = c.get("name", "")
            field_name = c.get("field_name", "")
            value = c.get("fact_value", "")
            if not name or not field_name or not value:
                continue
            q = format_infobox_question(name, field_name)
            questions.append(
                {
                    "id": f"q{len(questions)+1:03d}",
                    "question": q,
                    "gold_answer": clean_text(value),
                    "gold_chunk_ids": [c.get("chunk_id")],
                    "gold_context": c.get("text"),
                    "source_id": c.get("source_id"),
                    "source_url": c.get("url"),
                    "type": "infobox",
                }
            )

    return questions[:total_questions]


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build dataset + eval set from Wikipedia People NDJSON.")
    parser.add_argument("--input", nargs="+", required=True, help="Input NDJSON files")
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--eval-questions", type=int, default=100)
    parser.add_argument("--seed", type=str, default="people-demo")
    parser.add_argument("--out-people", required=True)
    parser.add_argument("--out-chunks", required=True)
    parser.add_argument("--out-eval", required=True)
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.input]
    records = read_ndjson(input_paths)
    sample = select_sample(records, args.sample_size, args.seed)

    out_people = Path(args.out_people)
    out_people.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_people, sample)

    chunks = build_chunks(sample)
    out_chunks = Path(args.out_chunks)
    out_chunks.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_chunks, chunks)

    eval_set = build_eval_set(chunks, args.eval_questions, seed=42, infobox_ratio=0.6)
    out_eval = Path(args.out_eval)
    out_eval.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_eval, eval_set)

    print(f"Wrote {len(sample)} people -> {out_people}")
    print(f"Wrote {len(chunks)} chunks -> {out_chunks}")
    print(f"Wrote {len(eval_set)} questions -> {out_eval}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
