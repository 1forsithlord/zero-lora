"""
LoRA trainer with optional preservation loss.
Usage: python -m zerolora.train --task dtd --loss raw --lambda 0.0 ...
"""
import os
# Fix for CUDA determinism
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import json
import time
import itertools
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .losses import get_loss, similarity_logits
from .frozen_eval import build_transform, env_info, load_task, seed_all
from .lora import ClipLoRA, load_clip
from .probes import ZS_DEV, class_embeddings, load_probe
from .prompts import prompts_for
from .refdata import ReferencePool, make_collate


class TrainConfig:
    def __init__(self, **kwargs):
        self.task = kwargs.get("task", "dtd")
        self.loss = kwargs.get("loss", "raw")
        self.lam = kwargs.get("lam", 0.0)
        self.rank = kwargs.get("rank", 4)
        self.alpha = kwargs.get("alpha", 8.0)
        self.epochs = kwargs.get("epochs", 5)
        self.lr = kwargs.get("lr", 1e-4)
        self.weight_decay = kwargs.get("weight_decay", 0.0)
        self.seed = kwargs.get("seed", 0)
        self.batch_size = kwargs.get("batch_size", 32)
        self.ref_batch_size = kwargs.get("ref_batch_size", 64)
        self.force_ref = kwargs.get("force_ref", False)
        self.max_steps = kwargs.get("max_steps", 0)


@torch.no_grad()
def evaluate(model, loader, class_text, device, return_preds=False):
    correct = 0
    n = 0
    all_preds = []
    all_labels = []
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model.logits(model.encode_image(x), class_text)
        preds = logits.argmax(-1)
        
        correct += (preds == y).sum().item()
        n += len(y)
        
        if return_preds:
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
            
    acc = correct / max(n, 1)
    if return_preds:
        return acc, torch.cat(all_preds), torch.cat(all_labels)
    return acc


def run_training(model, train_loader, dev_loader, probes, ref_loader, cfg, out_dir, device="cpu", extra=None):
    os.makedirs(out_dir, exist_ok=True)
    model.assert_frozen()
    loss_fn = get_loss(cfg.loss)
    
    use_ref = ref_loader is not None and (cfg.lam != 0.0 or cfg.force_ref)
    if cfg.lam != 0.0 and ref_loader is None:
        raise ValueError("lambda != 0 needs a reference pool (--ref-pool)")
        
    ref_iter = itertools.cycle(ref_loader) if use_ref else None
    opt = torch.optim.AdamW(model.lora_parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    metrics_file = open(f"{out_dir}/metrics.jsonl", "w")

    def log_metric(row):
        metrics_file.write(json.dumps(row) + "\n")
        metrics_file.flush()

    cuda = device != "cpu" and torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
        
    # Epoch 0 baseline (frozen CLIP)
    print("Evaluating frozen baseline...")
    frozen_acc = evaluate(model, dev_loader, model.class_text, device)
    frozen_zs = {name: evaluate(model, loader, ct, device) for name, (loader, ct) in probes.items()}
    frozen_zs_mean = sum(frozen_zs.values()) / len(frozen_zs) if frozen_zs else 0
    
    frozen_metrics = {"dev_task_acc": frozen_acc, "zs": frozen_zs, "zs_mean": frozen_zs_mean}
    log_metric({"epoch": 0, "step": 0, **frozen_metrics})
    
    step = 0
    t_train = 0.0
    n_samples = 0
    stopped_early = False
    
    for epoch in range(1, cfg.epochs + 1):
        if cuda: torch.cuda.synchronize()
        t0 = time.time()
        
        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            
            l_task = F.cross_entropy(model.forward_task((x,)), y)
            loss = l_task
            l_pres = None
            
            if use_ref:
                rb = next(ref_iter)
                rb = {k: v.to(device) for k, v in rb.items()}
                
                img_s, txt = model.forward_reference(rb)
                img_t = model.forward_teacher(rb)
                
                S_s = similarity_logits(img_s, txt, model.logit_scale)
                S_t = similarity_logits(img_t, txt, model.logit_scale)
                
                l_pres = loss_fn(S_s, S_t)
                loss = l_task + cfg.lam * l_pres
                
            if not torch.isfinite(loss):
                raise FloatingPointError(f"NaN/Inf loss at step {step}: {loss.item()}")
                
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            
            step += 1
            n_samples += len(y)
            
            if step % 10 == 0 or step == 1:
                log_metric({
                    "step": step, "epoch": epoch, 
                    "l_task": l_task.item(), 
                    "l_pres": l_pres.item() if l_pres else None, 
                    "loss": loss.item()
                })
                
            if cfg.max_steps > 0 and step >= cfg.max_steps:
                stopped_early = True
                break
                
        if cuda: torch.cuda.synchronize()
        t_train += time.time() - t0
        
        # Eval at end of epoch
        if epoch == cfg.epochs or stopped_early:
            print(f"Evaluating epoch {epoch}...")
            acc = evaluate(model, dev_loader, model.class_text, device)
            zs = {name: evaluate(model, loader, ct, device) for name, (loader, ct) in probes.items()}
            zs_mean = sum(zs.values()) / len(zs) if zs else 0
            
            final_metrics = {"dev_task_acc": acc, "zs": zs, "zs_mean": zs_mean}
            log_metric({"epoch": epoch, "step": step, **final_metrics})
            
        if stopped_early:
            break
            
    metrics_file.close()

    # Save adapter
    adapter_path = f"{out_dir}/adapter.pt"
    torch.save(model.adapter_state(), adapter_path)

    # Save predictions for Phase 5 statistical tests
    print("Saving per-image predictions for stat tests...")
    _, dev_preds, dev_labels = evaluate(model, dev_loader, model.class_text, device, return_preds=True)
    preds_dict = {"dev": {"preds": dev_preds, "labels": dev_labels}}
    
    for name, (loader, ct) in probes.items():
        _, p, l = evaluate(model, loader, ct, device, return_preds=True)
        preds_dict[name] = {"preds": p, "labels": l}
        
    torch.save(preds_dict, f"{out_dir}/predictions.pt")

    # Compile summary
    summary = {
        "config": vars(cfg),
        "steps": step,
        "frozen": frozen_metrics,
        "final": final_metrics,
        "delta_dev_task_acc": final_metrics["dev_task_acc"] - frozen_metrics["dev_task_acc"],
        "delta_zs": {k: final_metrics["zs"][k] - frozen_metrics["zs"][k] for k in final_metrics["zs"]},
        "delta_zs_mean": final_metrics["zs_mean"] - frozen_metrics["zs_mean"],
        "retention_ratio_zs_mean": final_metrics["zs_mean"] / frozen_metrics["zs_mean"] if frozen_metrics["zs_mean"] else 0,
        "train_time_s": round(t_train, 2),
        "peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if cuda else None,
        "trainable_params": model.num_trainable(),
    }
    
    if extra:
        summary.update(extra)
        
    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/protocol_v1.yaml")
    parser.add_argument("--root", default="data")
    parser.add_argument("--task", choices=["dtd", "eurosat"], required=True)
    parser.add_argument("--loss", choices=["raw", "zscl", "gnsp"], default="raw")
    parser.add_argument("--lambda", dest="lam", type=float, default=0.0)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ref-batch-size", type=int, default=64)
    parser.add_argument("--ref-pool", default=None)
    parser.add_argument("--force-reference-forward", dest="force_ref", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--probe-max-images", type=int, default=2000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--tag", default="")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    cfg_yaml = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set seed before loading model
    seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    print("Loading CLIP...")
    clip, tok, commit = load_clip(
        cfg_yaml["base_model"]["hf_id"], 
        cfg_yaml["base_model"].get("revision"), 
        device
    )
    
    # ensure we got the right commit
    want_rev = cfg_yaml["base_model"].get("revision")
    if want_rev and commit and commit != want_rev:
        raise ValueError(f"Commit mismatch: got {commit}, expected {want_rev}")

    # reset seed to make LoRA init deterministic
    seed_all(args.seed)
    
    alpha = args.alpha if args.alpha else 2.0 * args.rank
    model = ClipLoRA(clip, rank=args.rank, alpha=alpha).to(device)

    tf = build_transform(cfg_yaml["preprocessing"])
    datasets, classnames, hashes, sizes = load_task(args.task, cfg_yaml, args.root, tf)
    
    mode = cfg_yaml["prompts"]["mode_primary"]
    class_embeds = class_embeddings(model, tok, classnames, lambda c: prompts_for(args.task, c, mode), device)
    model.set_class_text(class_embeds)

    # DataLoaders
    g = torch.Generator(); g.manual_seed(args.seed)
    train_loader = DataLoader(datasets["adaptation"], batch_size=args.batch_size, shuffle=True, generator=g, num_workers=args.num_workers)
    dev_loader = DataLoader(datasets["dev"], batch_size=256, shuffle=False, num_workers=args.num_workers)

    # Probes
    probes = {}
    probe_meta = {}
    for name in ZS_DEV:
        p = load_probe(name, args.root, tf, args.probe_max_images, cfg_yaml["global_seed"])
        ct = class_embeddings(model, tok, p["classnames"], lambda c, p=p: [t.format(c) for t in p["templates"]], device)
        probes[name] = (DataLoader(p["dataset"], batch_size=256, shuffle=False, num_workers=args.num_workers), ct)
        probe_meta[name] = {"n": p["n"], "hash": p["hash"]}

    # Ref pool
    ref_loader = None
    pool_hash = None
    if args.ref_pool:
        pool = ReferencePool(args.ref_pool, tf)
        pool_hash = pool.pool_hash
        g2 = torch.Generator(); g2.manual_seed(args.seed + 10_000)
        ref_loader = DataLoader(
            pool, batch_size=args.ref_batch_size, shuffle=True, drop_last=True,
            generator=g2, collate_fn=make_collate(tok), num_workers=args.num_workers
        )

    cfg = TrainConfig(
        task=args.task, loss=args.loss, lam=args.lam, rank=args.rank, 
        alpha=alpha, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, 
        seed=args.seed, batch_size=args.batch_size, ref_batch_size=args.ref_batch_size, 
        force_ref=args.force_ref, max_steps=args.max_steps
    )

    run_id = f"{args.task}_{args.loss}_lam{args.lam:g}_r{args.rank}_s{args.seed}"
    if args.tag:
        run_id += f"_{args.tag}"
    out_dir = f"{args.out}/{run_id}"

    extra = {
        "run_id": run_id, "env": env_info(), "model_commit": commit, "pool_hash": pool_hash,
        "split_hashes": hashes, "split_sizes": sizes, "probes": probe_meta,
        "device": device, "test_pool_evaluated": False
    }

    print(f"Starting run: {run_id}")
    summary = run_training(model, train_loader, dev_loader, probes, ref_loader, cfg, out_dir, device, extra)
    
    # Print a quick summary to stdout
    print(f"\nDone. Task gain: {summary['delta_dev_task_acc']:+.4f} | ZS drop: {summary['delta_zs_mean']:+.4f}")


if __name__ == "__main__":
    main()