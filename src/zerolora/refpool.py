"""Reference-pool construction (Phase 2A): CC3M subset -> filtered, deduplicated, leakage-audited pool.

Library part (importable and unit-testable); CLI is scripts/build_reference_pool.py.

Filters, in cheap-first order (all counted in audit_log.jsonl):
  caption_empty -> class_name_caption -> corrupt_image -> nsfw -> dup_exact_pool -> dup_near_pool
  -> near_dup_vs_eval:<dataset>   (pHash Hamming <= thr against DTD/EuroSAT/ZS-probe images)

Caption matching is STEM-based (Porter), a deterministic, download-free and conservative superset of
lemma matching: 'forests' and 'forest' both match 'forest'. It over-removes slightly; that is the safe
direction for leakage. [Deviation from the literal "lemma-matched" wording: logged in the manifest.]
"""
import hashlib
import io
import json
import os
import re
import tarfile
import time
from collections import Counter

import numpy as np

from .phash import NearDupIndex, phash
from .poolutils import (compute_image_content_hash, compute_pool_hash, write_jsonl)
from .prompts import EUROSAT_CLASSNAMES

# Copied from results/run_manifest.json of the Phase 1 run (torchvision DTD class names).
DTD_CLASSES = [
    "banded", "blotchy", "braided", "bubbly", "bumpy", "chequered", "cobwebbed", "cracked", "crosshatched",
    "crystalline", "dotted", "fibrous", "flecked", "freckled", "frilly", "gauzy", "grid", "grooved",
    "honeycombed", "interlaced", "knitted", "lacelike", "lined", "marbled", "matted", "meshed", "paisley",
    "perforated", "pitted", "pleated", "polka-dotted", "porous", "potholed", "scaly", "smeared", "spiralled",
    "sprinkled", "stained", "stratified", "striped", "studded", "swirly", "veined", "waffled", "woven",
    "wrinkled", "zigzagged"]

_TOKEN = re.compile(r"[a-z]+")


def _stemmer():
    try:
        from nltk.stem import PorterStemmer
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install nltk (PorterStemmer needs no data download)") from e
    return PorterStemmer()


class ClassNameFilter:
    """Contiguous stemmed-token phrase matcher."""

    def __init__(self, phrases):
        self._st = _stemmer()
        self._cache = {}
        self.phrases = set()
        for p in phrases:
            toks = tuple(self._norm(t) for t in _TOKEN.findall(p.lower()))
            if toks:
                self.phrases.add(toks)
        self.maxlen = max(len(p) for p in self.phrases) if self.phrases else 1

    def _norm(self, t):
        if t not in self._cache:
            self._cache[t] = self._st.stem(t)
        return self._cache[t]

    def hit(self, caption):
        toks = [self._norm(t) for t in _TOKEN.findall(caption.lower())]
        for n in range(1, self.maxlen + 1):
            for i in range(len(toks) - n + 1):
                if tuple(toks[i:i + n]) in self.phrases:
                    return True
        return False


def expand_phrases(names):
    """'a or b or c' -> ['a','b','c'] (EuroSAT-style class names list alternatives)."""
    out = []
    for n in names:
        out.extend(x.strip() for x in n.split(" or ") if x.strip())
    return sorted(set(out))


def task_class_phrases():
    return sorted(set(expand_phrases(DTD_CLASSES) + expand_phrases(EUROSAT_CLASSNAMES)))


# ----------------------------------------------------------------------------- candidate sources
def _emit_wds(parts, key, shard):
    img = next((parts[e] for e in ("jpg", "jpeg", "png", "webp") if e in parts), None)
    cap, nsfw, meta = None, None, {}
    if "txt" in parts:
        cap = parts["txt"].decode("utf-8", "ignore").strip()
    if "json" in parts:
        try:
            meta = json.loads(parts["json"].decode("utf-8", "ignore"))
        except Exception:
            meta = {}
        cap = cap or (meta.get("caption") or "").strip()
        for k in ("nsfw", "punsafe", "NSFW"):
            if k in meta:
                nsfw = meta[k]
    if img is None or not cap:
        return {"bytes": None, "caption": cap or "", "key": key, "shard": shard, "nsfw": nsfw}
    return {"bytes": img, "caption": cap, "key": key, "shard": shard, "nsfw": nsfw}


def iter_wds_tar(path, shard=""):
    """Yield candidates from a WebDataset tar (members grouped by key prefix)."""
    cur, parts = None, {}
    with tarfile.open(path, "r") as tf:
        for m in tf:
            if not m.isfile():
                continue
            key, ext = os.path.splitext(os.path.basename(m.name))
            ext = ext.lower().lstrip(".")
            if key != cur:
                if parts:
                    yield _emit_wds(parts, cur, shard)
                cur, parts = key, {}
            parts[ext] = tf.extractfile(m).read()
    if parts:
        yield _emit_wds(parts, cur, shard)


def iter_pixparse(shard_ids, token=None, repo="pixparse/cc3m-wds"):
    """CC3M mirror with images included (no dead URLs). [VERIFY shard naming on first run]"""
    from huggingface_hub import hf_hub_download
    for sid in shard_ids:
        fn = f"cc3m-train-{int(sid):04d}.tar"
        path = hf_hub_download(repo_id=repo, filename=fn, repo_type="dataset", token=token)
        yield from iter_wds_tar(path, shard=fn)


def iter_urls(seed=1234, workers=32, timeout=5, chunk=256):
    """Original CC3M URL list (google-research-datasets/conceptual_captions). Many URLs are dead; dead
    ones are yielded as {'dead': True} so they are counted. UNTESTED in the authoring sandbox."""
    from concurrent.futures import ThreadPoolExecutor
    import requests
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/conceptual_captions", "unlabeled", split="train",
                      streaming=True).shuffle(seed=seed, buffer_size=10000)

    def fetch(row):
        try:
            r = requests.get(row["image_url"], timeout=timeout)
            if r.status_code == 200 and len(r.content) > 1000:
                return {"bytes": r.content, "caption": row["caption"].strip(), "key": row["image_url"],
                        "shard": "urls", "nsfw": None}
        except Exception:
            pass
        return {"dead": True}

    buf = []
    with ThreadPoolExecutor(workers) as ex:
        for row in ds:
            buf.append(row)
            if len(buf) >= chunk:
                yield from ex.map(fetch, buf)
                buf = []
        if buf:
            yield from ex.map(fetch, buf)


# ----------------------------------------------------------------------------- eval-image hash index
EVAL_DATASETS = ("dtd", "eurosat", "caltech101", "oxford_pets", "cifar100", "food101")


def _iter_eval_datasets(name, root):
    import torchvision.datasets as D
    if name == "dtd":
        for s in ("train", "val", "test"):
            yield f"dtd:{s}", D.DTD(root, split=s, partition=1, download=True)
    elif name == "eurosat":
        yield "eurosat:all", D.EuroSAT(root, download=True)
    elif name == "caltech101":
        yield "caltech101:all", D.Caltech101(root, download=True)
    elif name == "oxford_pets":
        for s in ("trainval", "test"):
            yield f"oxford_pets:{s}", D.OxfordIIITPet(root, split=s, download=True)
    elif name == "cifar100":
        for tr in (True, False):
            yield f"cifar100:{'train' if tr else 'test'}", D.CIFAR100(root, train=tr, download=True)
    elif name == "food101":
        for s in ("train", "test"):
            yield f"food101:{s}", D.Food101(root, split=s, download=True)
    else:
        raise KeyError(name)


def build_eval_index(root, names, thr=6, cache_dir=None, cap=0, seed=1234, log=print):
    """pHash every image of the named eval datasets (all splits). cap>0 -> seeded random sample per split
    (sampled check, recorded in `info` as exhaustive=False). Returns (NearDupIndex, info)."""
    idx = NearDupIndex(thr)
    info = {}
    for name in names:
        for tag, ds in _iter_eval_datasets(name, root):
            n = len(ds)
            sel = np.arange(n)
            if cap and cap < n:
                sel = np.sort(np.random.RandomState(seed).choice(n, cap, replace=False))
            cpath = None
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                cpath = os.path.join(cache_dir, f"{tag.replace(':', '_')}_cap{cap}_seed{seed}.npy")
            if cpath and os.path.exists(cpath):
                hs = np.load(cpath)
            else:
                t0 = time.time()
                hs = np.array([phash(ds[int(i)][0]) for i in sel], dtype=np.uint64)
                if cpath:
                    np.save(cpath, hs)
                log(f"  phash {tag}: {len(hs)}/{n} images in {time.time() - t0:.0f}s")
            idx.add_many((int(h) for h in hs), label=name)
            info[tag] = {"n_total": n, "n_hashed": int(len(hs)), "exhaustive": bool(len(hs) == n)}
    return idx, info


# ----------------------------------------------------------------------------- pipeline
def _nsfw_flag(v, thr=0.5):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v) > thr
    return str(v).strip().lower() in ("likely", "very_likely", "nsfw", "unsafe", "true", "1")


def run_pipeline(cands, out_dir, target, name_filter, eval_index, thr=6, probe_filter=None,
                 probe_mode="flag", time_limit_s=None, min_side=64, save_side=256, log=print):
    """Consume candidates until `target` are accepted (or sources/time run out)."""
    from PIL import Image
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    counts = Counter()
    pool_idx = NearDupIndex(thr)
    seen_sha = set()
    records, metadata = [], []
    t0 = time.time()
    stop = "source_exhausted"
    for cand in cands:
        if time_limit_s and time.time() - t0 > time_limit_s:
            stop = "time_limit"; break
        counts["candidates_seen"] += 1
        if cand.get("dead"):
            counts["url_unavailable"] += 1; continue
        cap = cand.get("caption", "")
        if len(cap) < 3 or cand.get("bytes") is None:
            counts["caption_empty_or_no_image"] += 1; continue
        if name_filter.hit(cap):
            counts["class_name_caption"] += 1; continue
        if probe_filter is not None and probe_filter.hit(cap):
            if probe_mode == "remove":
                counts["probe_class_name_caption"] += 1; continue
            counts["probe_class_name_flagged_kept"] += 1
        try:
            im = Image.open(io.BytesIO(cand["bytes"])); im.load(); im = im.convert("RGB")
            if min(im.size) < min_side:
                raise ValueError("too small")
        except Exception:
            counts["corrupt_image"] += 1; continue
        flag = _nsfw_flag(cand.get("nsfw"))
        if flag is None:
            counts["nsfw_field_missing"] += 1
        elif flag:
            counts["nsfw"] += 1; continue
        sha = hashlib.sha256(cand["bytes"]).hexdigest()
        if sha in seen_sha:
            counts["dup_exact_pool"] += 1; continue
        ph = phash(im)
        if pool_idx.query(ph) is not None:
            counts["dup_near_pool"] += 1; continue
        hit = eval_index.query(ph) if eval_index is not None else None
        if hit is not None:
            counts[f"near_dup_vs_eval:{hit[1]}"] += 1
            if hit[0] == 0:
                counts["near_dup_vs_eval:distance0"] += 1
            continue
        # accept
        seen_sha.add(sha); pool_idx.add(ph)
        i = len(records)
        rel = f"images/{i:06d}.jpg"
        w, h = im.size
        s = min(w, h)
        if s > save_side:
            f = save_side / s
            im = im.resize((max(1, round(w * f)), max(1, round(h * f))), Image.LANCZOS)
        im.save(os.path.join(out_dir, rel), "JPEG", quality=95)
        records.append({"path": rel, "caption": cap})
        metadata.append({"path": rel, "key": cand.get("key"), "shard": cand.get("shard"),
                         "src_sha256": sha, "phash": f"{ph:016x}"})
        counts["accepted"] += 1
        if len(records) % 500 == 0:
            log(f"  accepted {len(records)}/{target}  seen {counts['candidates_seen']}")
        if len(records) >= target:
            stop = "target_reached"; break
    return records, metadata, counts, stop, time.time() - t0


def finalize(out_dir, records, metadata, counts, manifest):
    write_jsonl(os.path.join(out_dir, "captions.jsonl"), records)
    write_jsonl(os.path.join(out_dir, "metadata.jsonl"), metadata)
    ph = compute_pool_hash(records)
    ih = compute_image_content_hash(out_dir, records)
    with open(os.path.join(out_dir, "pool_hash.json"), "w") as f:
        json.dump({"pool_hash_sha256": ph, "image_content_hash_sha256": ih, "n": len(records),
                   "definition": "sha256 over sorted (path, caption) tuples; image hash over sorted (path, sha256(file))"},
                  f, indent=2)
    rows = [{"filter": k, "count": int(v)} for k, v in sorted(counts.items())]
    rows.append({"summary": True, "accepted": len(records), "pool_hash": ph, **{k: manifest[k] for k in ("stopped_reason",)}})
    write_jsonl(os.path.join(out_dir, "audit_log.jsonl"), rows)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return ph, ih