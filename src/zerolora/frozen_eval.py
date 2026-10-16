"""Frozen CLIP zero-shot evaluation on DTD / EuroSAT (Phase 1, Day 3)."""
import argparse, csv, json, os, platform, random, subprocess, sys, time, zlib

import numpy as np
import yaml

from .prompts import class_names, prompts_for
from .splits import assert_disjoint, index_hash, stratified_split


def seed_all(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def env_info():
    import torch, torchvision, transformers
    info = {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "transformers": transformers.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    try:
        info["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        info["git_commit"] = None
    return info


def build_transform(p):
    import torchvision.transforms as T
    interp = {"bicubic": T.InterpolationMode.BICUBIC}[p["interpolation"]]
    return T.Compose([
        T.Resize(p["resize"], interpolation=interp), T.CenterCrop(p["center_crop"]),
        T.Lambda(lambda im: im.convert("RGB")), T.ToTensor(),
        T.Normalize(p["normalize_mean"], p["normalize_std"]),
    ])


def load_task(name, cfg, root, tf):
    import torchvision.datasets as D
    from torch.utils.data import Subset
    t = cfg["tasks"][name]
    if name == "dtd":
        part = t["partition"]
        base = {k: D.DTD(root, split=s, partition=part, transform=tf, download=True)
                for k, s in [("adaptation", t["adaptation"]), ("dev", t["dev"]), ("test", t["test"])]}
        ref = base["dev"]
        # FIX: hash file paths instead of index ranges
        pools = {k: np.array([zlib.crc32(str(p).encode()) for p in v._image_files], dtype=np.int64) for k, v in base.items()}
        datasets = base
        paths = {k: set(map(str, v._image_files)) for k, v in base.items()}
        assert not (paths["adaptation"] & paths["dev"]) and not (paths["adaptation"] & paths["test"]) \
            and not (paths["dev"] & paths["test"]), "DTD official pools overlap"
    elif name == "eurosat":
        full = D.EuroSAT(root, transform=tf, download=True)
        labels = np.array([y for _, y in full.samples]) if hasattr(full, "samples") else np.array(full.targets)
        pools = stratified_split(labels, t["fractions"], cfg["global_seed"])
        assert_disjoint(pools)
        datasets = {k: Subset(full, v.tolist()) for k, v in pools.items()}
        ref = full
    else:
        raise KeyError(name)
    hashes = {k: index_hash(v) for k, v in pools.items()}
    sizes = {k: len(v) for k, v in pools.items()}
    return datasets, class_names(name, ref), hashes, sizes


def run_once(cfg, root, device, batch_size, num_workers, eval_pools):
    import torch
    from torch.utils.data import DataLoader
    from transformers import CLIPModel, CLIPTokenizer

    mid = cfg["base_model"]["hf_id"]
    rev = cfg["base_model"].get("revision")
    model = CLIPModel.from_pretrained(mid, revision=rev).to(device).eval()
    tok = CLIPTokenizer.from_pretrained(mid, revision=rev)
    commit = rev if rev else getattr(model.config, "_commit_hash", None)
    tf = build_transform(cfg["preprocessing"])
    rows, preds_all, meta = [], {}, {"model_commit": commit, "tasks": {}}

    for name in cfg["tasks"]:
        if name not in ("dtd", "eurosat"):
            continue
        datasets, classnames, hashes, sizes = load_task(name, cfg, root, tf)
        meta["tasks"][name] = {"split_hashes": hashes, "sizes": sizes, "classnames": classnames}
        for mode in (cfg["prompts"]["mode_primary"], cfg["prompts"]["mode_secondary"]):
            with torch.no_grad():
                embs = []
                for c in classnames:
                    ids = tok(prompts_for(name, c, mode), padding="max_length", max_length=77,
                              truncation=True, return_tensors="pt").to(device)
                    text_outputs = model.text_model(**ids)
                    e = model.text_projection(text_outputs[1])
                    e = e / e.norm(dim=-1, keepdim=True)
                    e = e.mean(0); e = e / e.norm()
                    embs.append(e)
                text = torch.stack(embs)
                scale = model.logit_scale.exp().item()
            for pool in eval_pools:
                dl = DataLoader(datasets[pool], batch_size=batch_size, shuffle=False, num_workers=num_workers)
                preds, labels = [], []
                with torch.no_grad():
                    for x, y in dl:
                        vision_outputs = model.vision_model(pixel_values=x.to(device))
                        f = model.visual_projection(vision_outputs[1])
                        f = f / f.norm(dim=-1, keepdim=True)
                        preds.append((scale * f @ text.T).argmax(-1).cpu()); labels.append(y)
                preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
                acc = float((preds == labels).mean())
                per_cls = [float((preds[labels == k] == k).mean()) for k in range(len(classnames))]
                rows.append({"dataset": name, "pool": pool, "prompt_mode": mode, "n": len(labels),
                             "top1": round(acc, 6), "mean_per_class_acc": round(float(np.mean(per_cls)), 6),
                             "split_hash": hashes[pool], "model_id": mid, "model_commit": commit})
                preds_all[(name, pool, mode)] = preds
    return rows, preds_all, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/protocol_v1.yaml")
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--pools", nargs="+", default=["dev", "test"],
                    help="pools to evaluate; 'test' with a frozen model tunes nothing, but is logged")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--check-rerun", action="store_true")
    a = ap.parse_args()

    import torch
    cfg = yaml.safe_load(open(a.config))
    os.makedirs(a.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_all(cfg["global_seed"])
    t0 = time.time()
    rows, preds, meta = run_once(cfg, a.root, device, a.batch_size, a.num_workers, a.pools)
    meta.update({"env": env_info(), "device": device, "wall_clock_s": round(time.time() - t0, 1),
                 "global_seed": cfg["global_seed"], "preprocessing": cfg["preprocessing"],
                 "protocol_version": cfg["protocol_version"], "pools_evaluated": a.pools})

    if a.check_rerun:
        seed_all(cfg["global_seed"])
        rows2, preds2, _ = run_once(cfg, a.root, device, a.batch_size, a.num_workers, a.pools)
        same = all(np.array_equal(preds[k], preds2[k]) for k in preds)
        meta["rerun_predictions_identical"] = bool(same)
        print("GATE 1 rerun check:", "PASS (identical predictions)" if same else "FAIL (predictions differ)")
        if not same:
            sys.exit(1)

    with open(os.path.join(a.out, "baseline_frozen_clip.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(os.path.join(a.out, "run_manifest.json"), "w") as f:
        json.dump(meta, f, indent=2)
    for r in rows:
        print(r["dataset"], r["pool"], r["prompt_mode"], r["n"], r["top1"])


if __name__ == "__main__":
    main()