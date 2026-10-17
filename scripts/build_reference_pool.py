"""Build the leakage-audited CC3M reference pool (Phase 2A).

Colab:
    pip install -q nltk huggingface_hub
    python scripts/build_reference_pool.py --root /content/data --out data/reference_pool \
        --source pixparse --shard-ids 0 1 2 --target-size 8000

Outputs in --out: images/, captions.jsonl, pool_hash.json, audit_log.jsonl, metadata.jsonl, manifest.json.
Time box (protocol): if this takes more than ~3 days of effort, freeze the clean subset as-is and write the
limitations from manifest.json into the paper. The pool is built ONCE, its hash recorded, then frozen.
"""
import argparse
import json
import os
import pathlib
import platform
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from zerolora import refpool as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/reference_pool")
    ap.add_argument("--root", default="data", help="where torchvision datasets live/are downloaded")
    ap.add_argument("--source", choices=["pixparse", "urls"], default="pixparse")
    ap.add_argument("--shard-ids", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--target-size", type=int, default=8000, help="protocol range 5000-10000")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--hamming", type=int, default=6)
    ap.add_argument("--eval-datasets", nargs="+", default=list(R.EVAL_DATASETS),
                    help="datasets whose images the pool must not contain")
    ap.add_argument("--probe-max-images", type=int, default=0,
                    help="0 = hash every image (exhaustive). >0 = seeded sample per split (sampled check; recorded)")
    ap.add_argument("--probe-classname-mode", choices=["off", "flag", "remove"], default="flag",
                    help="what to do with captions that contain ZS-probe class names (checklist C)")
    ap.add_argument("--time-limit-min", type=float, default=240)
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--skip-eval-check", action="store_true", help="DEBUG ONLY: disables the disjointness check")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    names = R.task_class_phrases()
    name_filter = R.ClassNameFilter(names)
    probe_filter = None
    if a.probe_classname_mode != "off":
        print("NOTE: probe class names come from the torchvision datasets (downloads them if absent; Food101 ~5 GB).")
        probe_names = []
        import torchvision.datasets as D
        loaders = {"caltech101": lambda: [c.replace("_", " ") for c in D.Caltech101(a.root, download=True).categories],
                   "oxford_pets": lambda: list(D.OxfordIIITPet(a.root, download=True).classes),
                   "cifar100": lambda: list(D.CIFAR100(a.root, download=True).classes),
                   "food101": lambda: [c.replace("_", " ") for c in D.Food101(a.root, download=True).classes]}
        for k in a.eval_datasets:
            if k in loaders:
                try:
                    probe_names += loaders[k]()
                except Exception as e:  # keep going, but say so
                    print(f"WARNING: could not load class names of {k}:", e)
        probe_filter = R.ClassNameFilter(R.expand_phrases([p.lower() for p in probe_names])) if probe_names else None

    if a.skip_eval_check:
        print("WARNING: --skip-eval-check set; pool is NOT leakage-audited.")
        eval_index, eval_info = None, {}
    else:
        print("Hashing eval/probe images for the disjointness check ...")
        eval_index, eval_info = R.build_eval_index(
            a.root, a.eval_datasets, a.hamming, cache_dir=os.path.join(a.out, "..", "phash_cache"),
            cap=a.probe_max_images, seed=a.seed)

    cands = (R.iter_pixparse(a.shard_ids, a.hf_token) if a.source == "pixparse" else R.iter_urls(a.seed))
    records, meta, counts, stop, dt = R.run_pipeline(
        cands, a.out, a.target_size, name_filter, eval_index, a.hamming, probe_filter,
        a.probe_classname_mode, time_limit_s=a.time_limit_min * 60)

    manifest = {
        "source": ("pixparse/cc3m-wds (CC3M mirror with images; shards %s)" % a.shard_ids if a.source == "pixparse"
                   else "google-research-datasets/conceptual_captions URL list"),
        "args": vars(a) | {"hf_token": bool(a.hf_token)},
        "stopped_reason": stop, "n_accepted": len(records), "target": a.target_size,
        "target_reached": len(records) >= a.target_size,
        "eval_check": {"performed": not a.skip_eval_check, "datasets": eval_info, "hamming_thr": a.hamming},
        "caption_filter": {"method": "Porter-stem contiguous phrase match (conservative superset of lemma match)",
                           "n_phrases": len(name_filter.phrases),
                           "probe_classnames": a.probe_classname_mode},
        "image_storage": "JPEG q95, shorter side capped at 256 px (no upscaling); dedup/pHash computed on originals",
        "nsfw": "dropped only if the source provides a flag; see 'nsfw_field_missing' count",
        "limitations": [
            "CLIP's own pretraining data may overlap CC3M-like web data; not checkable here.",
            "near-duplicate check is pHash (Hamming<=%d): misses heavy crops/edits." % a.hamming,
            "eval-image check is exhaustive only where eval_check.datasets[*].exhaustive is true.",
            "shards used are a deterministic choice, not a random draw from all of CC3M."],
        "wall_clock_s": round(time.time() - t0, 1), "python": platform.python_version(),
    }
    ph, ih = R.finalize(a.out, records, meta, counts, manifest)
    print(json.dumps({"accepted": len(records), "stopped": stop, "pool_hash": ph, "image_hash": ih,
                      "counts": dict(counts)}, indent=2))
    if len(records) < 5000:
        print("WARNING: pool smaller than the protocol minimum (5000); add shards (--shard-ids) and rerun.")


if __name__ == "__main__":
    main()