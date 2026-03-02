#!/usr/bin/env python3
"""Build a UK aseptic cleanroom RAG corpus from local PDFs and generate eval questions.

Inputs:
  - data/pdf/*.pdf (EU GMP Annex 1, MHRA Specials guidance, etc.)
Outputs:
  - data/processed/documents.jsonl
  - data/processed/chunks.jsonl
  - eval/questions_100.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

# Ensure repo root is on sys.path when running as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.pipeline_defaults import (
    DEFAULT_CHUNK_MAX_CHARS,
    DEFAULT_CHUNK_OVERLAP_PARAGRAPHS,
    DEFAULT_CHUNK_TARGET_CHARS,
    DEFAULT_DATASET_SEED,
    DEFAULT_EVAL_QUESTIONS,
    DEFAULT_PDF_INPUT_DIR,
    EVAL_REPEAT_CHUNK_SKIP_PROB,
    EVAL_SENTENCE_MAX_CHARS,
    EVAL_SENTENCE_MIN_CHARS,
    EVAL_TOP_SENTENCES_PER_CHUNK,
    HEADING_MAX_CHARS,
    HEADING_MAX_WORDS,
    PARAGRAPH_MIN_CHARS,
    TABLELIKE_LINE_BLOCK_MAX_CHARS,
)

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
HEADING_NUM_RE = re.compile(r"^(\d+(?:\.\d+){0,4})\s+(.+)$")
STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "for",
    "on",
    "by",
    "with",
    "that",
    "this",
    "is",
    "are",
    "be",
    "as",
    "at",
    "from",
    "it",
    "its",
    "their",
    "these",
    "those",
    "should",
    "must",
    "may",
    "can",
    "will",
}

# Known source metadata for local UK aseptic cleanroom corpus.
DOC_CATALOG: dict[str, dict[str, str]] = {
    "20220825_gmp-an1_en_0.pdf": {
        "title": "EU GMP Annex 1: Manufacture of Sterile Medicinal Products",
        "url": "https://health.ec.europa.eu/document/download/e05af55b-38e9-42bf-8495-194bbf0b9262_en?filename=20220825_gmp-an1_en_0.pdf",
        "publisher": "European Commission",
        "jurisdiction": "UK-applicable / EU GMP",
        "doc_type": "guideline",
        "short_name": "EU GMP Annex 1",
    },
    "QA_Version_3_-_Aseptic_manip_updates.pdf": {
        "title": "MHRA Guidance for 'Specials' Manufacturers",
        "url": "https://assets.publishing.service.gov.uk/media/603526f28fa8f54330a8e25f/QA_Version_3_-_Aseptic_manip_updates.pdf",
        "publisher": "MHRA (UK)",
        "jurisdiction": "UK",
        "doc_type": "guidance",
        "short_name": "MHRA Specials Guidance",
    },
}


def stable_hash(value: str, seed: str) -> int:
    h = hashlib.blake2b(f"{seed}:{value}".encode("utf-8"), digest_size=8)
    return int.from_bytes(h.digest(), "big")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_cmd(args: list[str]) -> str:
    proc = subprocess.run(args, check=True, capture_output=True, text=True)
    return proc.stdout


def pdf_page_count(pdf_path: Path) -> int:
    info = run_cmd(["pdfinfo", str(pdf_path)])
    for line in info.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError(f"Could not parse page count for {pdf_path}")


def extract_pdf_pages(pdf_path: Path) -> list[str]:
    text = run_cmd(["pdftotext", "-layout", "-enc", "UTF-8", str(pdf_path), "-"])
    pages = text.split("\f")
    if pages and not pages[-1].strip():
        pages = pages[:-1]
    return pages


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "doc"


def is_noise_line(line: str) -> bool:
    s = normalize_space(line)
    if not s:
        return True
    if re.fullmatch(r"\d+", s):
        return True
    if s in {"EN", "EUROPEAN", "COMMISSION"}:
        return True
    if s.startswith("Revision") and len(s) < 50:
        return True
    if s.startswith("Page ") and len(s) < 20:
        return True
    if s.startswith("MHRA Guidance for Specials Manufacturers"):
        return True
    return False


def looks_like_heading(line: str) -> bool:
    s = normalize_space(line)
    if not s or len(s) > HEADING_MAX_CHARS:
        return False
    if is_noise_line(s):
        return False
    if s.endswith(".") and not HEADING_NUM_RE.match(s):
        return False
    if re.match(r"^\d+(?:\.\d+){0,4}\s+", s):
        return True
    words = s.split()
    if len(words) > HEADING_MAX_WORDS:
        return False
    alpha_words = [w for w in words if re.search(r"[A-Za-z]", w)]
    if not alpha_words:
        return False
    upper_ratio = sum(1 for w in alpha_words if w.isupper()) / max(1, len(alpha_words))
    titleish = sum(1 for w in alpha_words if w[:1].isupper()) / max(1, len(alpha_words))
    if upper_ratio >= 0.7:
        return True
    if titleish >= 0.8 and not any(s.lower().startswith(p) for p in ("the purpose", "this guidance", "however,")):
        return True
    return False


def join_lines(lines: list[str]) -> str:
    out: list[str] = []
    for raw in lines:
        line = normalize_space(raw)
        if not line:
            continue
        if not out:
            out.append(line)
            continue
        prev = out[-1]
        if prev.endswith("-") and line[:1].islower():
            out[-1] = prev[:-1] + line
            continue
        if re.search(r"[,:;(]$", prev) or (not re.search(r"[.!?]$", prev) and line[:1].islower()):
            out[-1] = prev + " " + line
        else:
            out[-1] = prev + " " + line
    return normalize_space(" ".join(out))


def extract_paragraphs(pages: list[str]) -> list[dict[str, Any]]:
    paragraphs: list[dict[str, Any]] = []
    current_heading: str | None = None

    for page_idx, page_text in enumerate(pages, start=1):
        lines = page_text.splitlines()
        buf: list[str] = []

        def flush_buf() -> None:
            nonlocal buf
            text = join_lines(buf)
            buf = []
            if not text:
                return
            if len(text) < PARAGRAPH_MIN_CHARS:
                return
            # Drop table-like rows with sparse columns.
            if text.count("  ") >= 3 and len(text) < TABLELIKE_LINE_BLOCK_MAX_CHARS:
                return
            paragraphs.append(
                {
                    "page": page_idx,
                    "heading": current_heading,
                    "text": text,
                }
            )

        for line in lines:
            stripped = line.strip()
            if not stripped:
                flush_buf()
                continue
            if is_noise_line(stripped):
                continue
            if looks_like_heading(stripped):
                flush_buf()
                current_heading = normalize_space(stripped)
                continue
            buf.append(stripped)

        flush_buf()

    return paragraphs


def chunk_paragraphs(
    paragraphs: list[dict[str, Any]],
    *,
    source_id: str,
    title: str,
    url: str,
    target_chars: int,
    max_chars: int,
    overlap_paragraphs: int,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    i = 0
    chunk_idx = 0

    while i < len(paragraphs):
        start = i
        cur: list[dict[str, Any]] = []
        cur_len = 0
        while i < len(paragraphs):
            p = paragraphs[i]
            p_len = len(p["text"])
            if cur and (cur_len + p_len > max_chars):
                break
            cur.append(p)
            cur_len += p_len + 1
            i += 1
            if cur_len >= target_chars:
                break

        if not cur:
            i += 1
            continue

        page_start = cur[0]["page"]
        page_end = cur[-1]["page"]
        heading = next((p.get("heading") for p in cur if p.get("heading")), None)
        if not heading:
            heading = f"Pages {page_start}-{page_end}" if page_start != page_end else f"Page {page_start}"

        body = "\n\n".join(p["text"] for p in cur)
        retrieval_text = (
            f"Document: {title}\n"
            f"Section: {heading}\n"
            f"Pages: {page_start}-{page_end}\n"
            f"Content: {body}"
        )

        chunks.append(
            {
                "chunk_id": f"{source_id}:chunk:{chunk_idx:04d}",
                "source_id": source_id,
                "name": title,
                "url": url,
                "section": heading,
                "type": "pdf_chunk",
                "text": retrieval_text,
                "doc_title": title,
                "page_start": page_start,
                "page_end": page_end,
                "heading": heading,
            }
        )
        chunk_idx += 1

        if i >= len(paragraphs):
            break
        if overlap_paragraphs > 0:
            i = max(start + 1, i - overlap_paragraphs)

    return chunks


def sentence_candidates(chunk: dict[str, Any]) -> list[str]:
    text = str(chunk.get("text") or "")
    # Keep only content after metadata prefix for eval answers.
    if "Content:" in text:
        text = text.split("Content:", 1)[1].strip()

    out: list[str] = []
    for sentence in SENT_SPLIT_RE.split(text):
        s = normalize_space(sentence)
        if len(s) < EVAL_SENTENCE_MIN_CHARS or len(s) > EVAL_SENTENCE_MAX_CHARS:
            continue
        if not re.search(r"[A-Za-z]{4}", s):
            continue
        if s.lower().startswith(("copyright", "all rights reserved")):
            continue
        out.append(s)
    return out


def score_sentence(s: str) -> int:
    t = s.lower()
    score = 0
    for term, pts in [
        ("environmental monitoring", 6),
        ("process monitoring", 5),
        ("contamination", 4),
        ("aseptic", 4),
        ("cleanroom", 4),
        ("hepa", 5),
        ("deviation", 3),
        ("trending", 3),
        ("quality", 2),
        ("should", 2),
        ("must", 2),
        ("expect", 2),
    ]:
        if term in t:
            score += pts
    score += min(4, t.count(";") + t.count(":"))
    return score


def topic_from_chunk(chunk: dict[str, Any], sentence: str) -> str:
    heading = normalize_space(str(chunk.get("heading") or chunk.get("section") or ""))
    if heading and not heading.lower().startswith("page "):
        heading = re.sub(r"^\d+(?:\.\d+){0,4}\s+", "", heading).strip()
        return heading[:90]

    words = [w for w in re.findall(r"[A-Za-z0-9'-]+", sentence) if w.lower() not in STOPWORDS]
    return " ".join(words[:8]) or "this requirement"


def build_question(doc_short: str, chunk: dict[str, Any], sentence: str, rnd: random.Random) -> str:
    topic = topic_from_chunk(chunk, sentence)
    templates = [
        f"According to {doc_short}, what does the guidance say about {topic}?",
        f"What is expected in {doc_short} regarding {topic}?",
        f"In {doc_short}, what requirement is described for {topic}?",
    ]
    page_start = chunk.get("page_start")
    page_end = chunk.get("page_end")
    if isinstance(page_start, int):
        page_label = str(page_start) if page_start == page_end else f"pages {page_start}-{page_end}"
        templates.append(f"What does {doc_short} state about {topic} ({page_label})?")
    return rnd.choice(templates)


def build_eval_set(chunks: list[dict[str, Any]], total_questions: int, seed: int) -> list[dict[str, Any]]:
    rnd = random.Random(seed)
    candidates: list[tuple[int, int, dict[str, Any], str]] = []

    for idx, c in enumerate(chunks):
        sentences = sentence_candidates(c)
        if not sentences:
            continue
        # Keep top 1-2 sentences per chunk to maintain variety.
        ranked = sorted(sentences, key=score_sentence, reverse=True)[:EVAL_TOP_SENTENCES_PER_CHUNK]
        for s in ranked:
            candidates.append((score_sentence(s), idx, c, s))

    # Stable random tie-break after scoring.
    candidates.sort(key=lambda x: (-x[0], stable_hash(f"{x[2]['chunk_id']}:{x[3]}", str(seed))))

    selected: list[dict[str, Any]] = []
    used_chunks: set[str] = set()

    for _, _, c, sentence in candidates:
        if len(selected) >= total_questions:
            break
        chunk_id = str(c.get("chunk_id"))
        if chunk_id in used_chunks and rnd.random() < EVAL_REPEAT_CHUNK_SKIP_PROB:
            continue
        doc_short = DOC_CATALOG.get(Path(str(c.get("url") or "")).name, {}).get("short_name") or str(c.get("name") or "the guidance")
        q = build_question(doc_short, c, sentence, rnd)
        selected.append(
            {
                "id": f"q{len(selected)+1:03d}",
                "question": q,
                "gold_answer": sentence,
                "gold_chunk_ids": [chunk_id],
                "gold_context": c.get("text"),
                "source_id": c.get("source_id"),
                "source_url": c.get("url"),
                "doc_title": c.get("name"),
                "section": c.get("section"),
                "page_start": c.get("page_start"),
                "page_end": c.get("page_end"),
                "type": "regulatory_clause",
            }
        )
        used_chunks.add(chunk_id)

    if len(selected) < total_questions:
        # Fallback: use first sentence from remaining chunks.
        for c in chunks:
            if len(selected) >= total_questions:
                break
            chunk_id = str(c.get("chunk_id"))
            if any(chunk_id in x.get("gold_chunk_ids", []) for x in selected):
                continue
            sents = sentence_candidates(c)
            if not sents:
                continue
            sentence = sents[0]
            doc_short = str(c.get("name") or "the guidance")
            selected.append(
                {
                    "id": f"q{len(selected)+1:03d}",
                    "question": f"What does {doc_short} say about {topic_from_chunk(c, sentence)}?",
                    "gold_answer": sentence,
                    "gold_chunk_ids": [chunk_id],
                    "gold_context": c.get("text"),
                    "source_id": c.get("source_id"),
                    "source_url": c.get("url"),
                    "doc_title": c.get("name"),
                    "section": c.get("section"),
                    "page_start": c.get("page_start"),
                    "page_end": c.get("page_end"),
                    "type": "regulatory_clause",
                }
            )

    return selected[:total_questions]


def infer_doc_metadata(pdf_path: Path, page_count: int) -> dict[str, Any]:
    catalog = DOC_CATALOG.get(pdf_path.name, {})
    title = catalog.get("title") or pdf_path.stem.replace("_", " ")
    url = catalog.get("url") or f"file://{pdf_path}"
    publisher = catalog.get("publisher") or "Unknown"
    jurisdiction = catalog.get("jurisdiction") or "UK"
    doc_type = catalog.get("doc_type") or "guidance"

    source_id = slugify(pdf_path.stem)
    return {
        "source_id": source_id,
        "name": title,
        "url": url,
        "raw": {
            "file_name": pdf_path.name,
            "file_path": str(pdf_path),
            "page_count": page_count,
            "publisher": publisher,
            "jurisdiction": jurisdiction,
            "doc_type": doc_type,
            "dataset": "uk-aseptic-cleanroom-pdf",
        },
    }


def build_corpus(
    pdf_paths: list[Path],
    *,
    target_chars: int,
    max_chars: int,
    overlap_paragraphs: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    documents: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []

    for pdf_path in pdf_paths:
        page_count = pdf_page_count(pdf_path)
        pages = extract_pdf_pages(pdf_path)
        if len(pages) != page_count:
            # pdftotext occasionally differs by trailing empty page markers; keep observed count for parsing.
            page_count = len(pages)

        doc = infer_doc_metadata(pdf_path, page_count)
        documents.append(doc)

        paragraphs = extract_paragraphs(pages)
        doc_chunks = chunk_paragraphs(
            paragraphs,
            source_id=doc["source_id"],
            title=doc["name"],
            url=doc["url"],
            target_chars=target_chars,
            max_chars=max_chars,
            overlap_paragraphs=overlap_paragraphs,
        )

        # Attach dataset metadata fields onto each chunk for ingestion.
        for c in doc_chunks:
            c["publisher"] = doc["raw"]["publisher"]
            c["jurisdiction"] = doc["raw"]["jurisdiction"]
            c["dataset"] = doc["raw"]["dataset"]
            c["doc_type"] = doc["raw"]["doc_type"]
            c["file_name"] = doc["raw"]["file_name"]

        chunks.extend(doc_chunks)

    return documents, chunks


def main() -> int:
    parser = argparse.ArgumentParser(description="Build UK aseptic cleanroom PDF corpus + eval set")
    parser.add_argument("--input-dir", default=DEFAULT_PDF_INPUT_DIR, help="Directory containing source PDFs")
    parser.add_argument("--input", nargs="*", help="Optional explicit PDF file list (overrides --input-dir)")
    parser.add_argument("--eval-questions", type=int, default=DEFAULT_EVAL_QUESTIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_DATASET_SEED)
    parser.add_argument("--chunk-target-chars", type=int, default=DEFAULT_CHUNK_TARGET_CHARS)
    parser.add_argument("--chunk-max-chars", type=int, default=DEFAULT_CHUNK_MAX_CHARS)
    parser.add_argument("--chunk-overlap-paragraphs", type=int, default=DEFAULT_CHUNK_OVERLAP_PARAGRAPHS)
    parser.add_argument("--out-docs", default="data/processed/documents.jsonl")
    parser.add_argument("--out-chunks", default="data/processed/chunks.jsonl")
    parser.add_argument("--out-eval", default="eval/questions_100.jsonl")
    args = parser.parse_args()

    if args.input:
        pdf_paths = [Path(p) for p in args.input]
    else:
        pdf_paths = sorted(Path(args.input_dir).glob("*.pdf"))

    if not pdf_paths:
        raise SystemExit("No PDF inputs found")

    documents, chunks = build_corpus(
        pdf_paths,
        target_chars=args.chunk_target_chars,
        max_chars=args.chunk_max_chars,
        overlap_paragraphs=args.chunk_overlap_paragraphs,
    )
    eval_set = build_eval_set(chunks, args.eval_questions, seed=args.seed)

    write_jsonl(Path(args.out_docs), documents)
    write_jsonl(Path(args.out_chunks), chunks)
    write_jsonl(Path(args.out_eval), eval_set)

    print(f"Wrote {len(documents)} documents -> {args.out_docs}")
    print(f"Wrote {len(chunks)} chunks -> {args.out_chunks}")
    print(f"Wrote {len(eval_set)} questions -> {args.out_eval}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
