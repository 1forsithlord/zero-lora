"""Minimal LoRA trainer with an optional preservation loss (Phase 2B).

    python -m zerolora.train --task dtd --loss raw --lambda 0.0 --rank 4 --epochs 5 --lr 1e-4 --seed 0 --out results
    (from repo root: PYTHONPATH=src python -m zerolora.train ... or scripts/run_train.py ...)

L_train = CE(task logits) + lambda * L_pres(S_student, S_teacher),  S = logit_scale * cos(img, txt) on a
REFERENCE batch (N images x N captions). Teacher = same network with LoRA off, no grad.
--lambda 0 with --loss X and no --force-reference-forward never touches the reference pool => plain LoRA.
With --force-reference-forward the reference path runs but is multiplied by 0.0 (used by the bit-for-bit test).

Never evaluates the held-out TEST pool or ZS-test probes (those belong to the final evaluation script).
Defaults for lr/epochs/alpha/batch are pilot placeholders, NOT tuned; tune on DEV only.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # must precede CUDA init (determinism)

import argparse
import dataclasses
import json
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .losses import get_loss, similarity_logits


@dataclasses.dataclass
class TrainConfig:
    task: str = "dtd"
    loss: str = "raw"
    lam: float = 0.0
    rank: int = 4
    alpha: float = 8.0
    epochs: int = 5
    lr: float = 1e-4
    weight_decay: float = 0.0
    seed: int = 0
    batch_size: int = 32
    ref_batch_size: int = 64
    force_ref: bool = False
    max_steps: int = 0          # 0 = no cap (use for timing pilots)
    log_every: int = 10
    eval_every: int = 1


def _cycle(loader):
    while True:
        for b in loader:
            yield b


def _to(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def evaluate_classifier(model, loader, class_text, device):
    correct = n = 0
    for x, y in loader:
        pred = model.logits(model.encode_image(x.to(device)), class_text).argmax(-1).cpu()
        correct += int((pred == y).sum()); n += len(y)
    return correct / max(n, 1)


def evaluate_all(model, dev_loader, probes, device):
    out = {"dev_task_acc": evaluate_classifier(model, dev_loader, model.class_text, device), "zs": {}}
    for name, (loader, ct) in probes.items():
        out["zs"][name] = evaluate_classifier(model, loader, ct, device)
    out["zs_mean"] = sum(out["zs"].values()) / len(out["zs"]) if out["zs"] else None
    return out


def run_training(model, train_loader, dev_loader, probes, ref_loader, cfg: TrainConfig, out_dir, device="cpu",
                 extra_summary=None):
    """Core loop (dependency-injected so it can be tested offline). Returns the summary dict."""
    os.makedirs(out_dir, exist_ok=True)
    model.assert_frozen()
    loss_fn = get_loss(cfg.loss)
    use_ref = ref_loader is not None and (cfg.lam != 0.0 or cfg.force_ref)
    if cfg.lam != 0.0 and ref_loader is None:
        raise ValueError("lambda != 0 needs a reference pool (--ref-pool)")
    ref_iter = _cycle(ref_loader) if use_ref else None
    opt = torch.optim.AdamW(model.lora_parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    mpath = os.path.join(out_dir, "metrics.jsonl")
    mf = open(mpath, "w")

    def log(row):
        mf.write(json.dumps(row) + "\n"); mf.flush()

    cuda = device != "cpu" and torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    frozen = evaluate_all(model, dev_loader, probes, device)      # epoch 0: LoRA B=0 == frozen CLIP
    log({"epoch": 0, "step": 0, **frozen})
    step, t_train, n_samples, stop = 0, 0.0, 0, False
    history, last = [], frozen
    for epoch in range(1, cfg.epochs + 1):
        if cuda: torch.cuda.synchronize()
        t0 = time.perf_counter()
        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            l_task = F.cross_entropy(model.forward_task((x,)), y)
            loss, l_pres = l_task, None
            if use_ref:
                rb = _to(next(ref_iter), device)
                img_s, txt = model.forward_reference(rb)
                img_t = model.forward_teacher(rb)
                S_s = similarity_logits(img_s, txt, model.logit_scale)
                S_t = similarity_logits(img_t, txt, model.logit_scale)
                l_pres = loss_fn(S_s, S_t)
                loss = l_task + cfg.lam * l_pres
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}: {loss.item()}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1; n_samples += len(y)
            if step % cfg.log_every == 0 or step == 1:
                row = {"step": step, "epoch": epoch, "l_task": l_task.item(),
                       "l_pres": None if l_pres is None else l_pres.item(), "loss": loss.item()}
                history.append(row); log(row)
            if cfg.max_steps and step >= cfg.max_steps:
                stop = True; break
        if cuda: torch.cuda.synchronize()
        t_train += time.perf_counter() - t0
        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs or stop:
            last = evaluate_all(model, dev_loader, probes, device)
            log({"epoch": epoch, "step": step, **last})
        if stop:
            break
    mf.close()

    ad = model.adapter_state()
    apath = os.path.join(out_dir, "adapter.pt")
    torch.save(ad, apath)
    summary = {
        "config": dataclasses.asdict(cfg), "steps": step, "epochs_run": epoch,
        "frozen": frozen, "final": last,
        "delta_dev_task_acc": last["dev_task_acc"] - frozen["dev_task_acc"],
        "delta_zs": {k: last["zs"][k] - frozen["zs"][k] for k in last["zs"]},
        "delta_zs_mean": None if last["zs_mean"] is None else last["zs_mean"] - frozen["zs_mean"],
        # retention ratio is reported ONLY next to the absolute numbers above (roadmap rule)
        "retention_ratio_zs_mean": None if not frozen["zs_mean"] else last["zs_mean"] / frozen["zs_mean"],
        "train_time_s": round(t_train, 2), "samples_per_s": round(n_samples / max(t_train, 1e-9), 2),
        "peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if cuda else None,
        "trainable_params": model.num_trainable(), "adapter_bytes": os.path.getsize(apath),
        "logit_scale": model.logit_scale, "used_reference_path": bool(use_ref),
    }
    if extra_summary:
        summary.update(extra_summary)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/protocol_v1.yaml")
    ap.add_argument("--root", default="data")
    ap.add_argument("--task", choices=["dtd", "eurosat"], required=True)
    ap.add_argument("--loss", choices=["raw", "zscl", "gnsp"], default="raw")
    ap.add_argument("--lambda", dest="lam", type=float, default=0.0)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=None, help="default 2*rank")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--ref-batch-size", type=int, default=64)
    ap.add_argument("--ref-pool", default=None, help="data/reference_pool")
    ap.add_argument("--force-reference-forward", dest="force_ref", action="store_true")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--probe-max-images", type=int, default=2000)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="results")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    import yaml
    from .frozen_eval import build_transform, env_info, load_task, seed_all
    from .lora import ClipLoRA, load_clip
    from .probes import ZS_DEV, class_embeddings, load_probe
    from .prompts import prompts_for
    from .refdata import ReferencePool, make_collate

    cfg_y = yaml.safe_load(open(a.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_all(a.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    clip, tok, commit = load_clip(cfg_y["base_model"]["hf_id"], cfg_y["base_model"].get("revision"), device)
    want = cfg_y["base_model"].get("revision")
    if want and commit and commit != want:
        raise RuntimeError(f"model commit {commit} != pinned {want}")
    alpha = a.alpha if a.alpha is not None else 2.0 * a.rank
    seed_all(a.seed)  # LoRA A init must not depend on model-loading RNG use
    model = ClipLoRA(clip, rank=a.rank, alpha=alpha).to(device)

    tf = build_transform(cfg_y["preprocessing"])
    datasets, classnames, hashes, sizes = load_task(a.task, cfg_y, a.root, tf)
    mode = cfg_y["prompts"]["mode_primary"]
    model.set_class_text(class_embeddings(model, tok, classnames, lambda c: prompts_for(a.task, c, mode), device))
    g = torch.Generator(); g.manual_seed(a.seed)
    train_loader = DataLoader(datasets["adaptation"], batch_size=a.batch_size, shuffle=True, generator=g,
                              num_workers=a.num_workers)
    dev_loader = DataLoader(datasets["dev"], batch_size=256, shuffle=False, num_workers=a.num_workers)

    probes, probe_meta = {}, {}
    for name in ZS_DEV:
        p = load_probe(name, a.root, tf, a.probe_max_images, cfg_y["global_seed"])
        ct = class_embeddings(model, tok, p["classnames"], lambda c, p=p: [t.format(c) for t in p["templates"]], device)
        probes[name] = (DataLoader(p["dataset"], batch_size=256, shuffle=False, num_workers=a.num_workers), ct)
        probe_meta[name] = {"n": p["n"], "hash": p["hash"]}

    ref_loader, pool_hash = None, None
    if a.ref_pool:
        pool = ReferencePool(a.ref_pool, tf)
        pool_hash = pool.pool_hash
        g2 = torch.Generator(); g2.manual_seed(a.seed + 10_000)   # separate stream: reference sampling never
        ref_loader = DataLoader(pool, batch_size=a.ref_batch_size, shuffle=True, drop_last=True,  # perturbs the task stream
                                generator=g2, collate_fn=make_collate(tok), num_workers=a.num_workers)

    cfg = TrainConfig(task=a.task, loss=a.loss, lam=a.lam, rank=a.rank, alpha=alpha, epochs=a.epochs, lr=a.lr,
                      weight_decay=a.weight_decay, seed=a.seed, batch_size=a.batch_size,
                      ref_batch_size=a.ref_batch_size, force_ref=a.force_ref, max_steps=a.max_steps)
    run_id = f"{a.task}_{a.loss}_lam{a.lam:g}_r{a.rank}_s{a.seed}" + (f"_{a.tag}" if a.tag else "")
    out_dir = os.path.join(a.out, run_id)
    extra = {"run_id": run_id, "env": env_info(), "model_commit": commit, "pool_hash": pool_hash,
             "split_hashes": hashes, "split_sizes": sizes, "probes": probe_meta,
             "device": device, "test_pool_evaluated": False, "zs_test_probes_evaluated": False}
    s = run_training(model, train_loader, dev_loader, probes, ref_loader, cfg, out_dir, device, extra)
    print(json.dumps({k: s[k] for k in ("run_id", "steps", "frozen", "final", "delta_dev_task_acc",
                                        "delta_zs_mean", "train_time_s", "peak_gpu_mem_mb")}, indent=2))


if __name__ == "__main__":
    main()