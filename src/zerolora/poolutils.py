"""Stdlib-only helpers shared by the reference-pool builder and the loader."""
import hashlib
import json
import os


def compute_pool_hash(records):
    """SHA256 over sorted (path, caption) tuples, as specified in the Phase 2 request."""
    h = hashlib.sha256()
    for p, c in sorted((r["path"], r["caption"]) for r in records):
        h.update(json.dumps([p, c], ensure_ascii=False).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def compute_image_content_hash(pool_dir, records):
    """SHA256 over sorted (path, sha256(file bytes)): detects a changed image behind an unchanged path."""
    h = hashlib.sha256()
    for r in sorted(records, key=lambda r: r["path"]):
        with open(os.path.join(pool_dir, r["path"]), "rb") as f:
            fh = hashlib.sha256(f.read()).hexdigest()
        h.update(json.dumps([r["path"], fh]).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def read_captions(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")