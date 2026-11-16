import argparse
import os
import sys
import pathlib
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from zerolora.lora import ClipLoRA, load_clip
from zerolora.probes import class_embeddings, load_probe, ZS_TEST
from zerolora.final_prompts import CIFAR100_TEMPLATES, FOOD101_TEMPLATES
from zerolora.frozen_eval import build_transform
import yaml

@torch.no_grad()
def evaluate(model, loader, class_text, device):
    correct = n = 0
    for x, y in loader:
        pred = model.logits(model.encode_image(x.to(device)), class_text).argmax(-1).cpu()
        correct += int((pred == y).sum())
        n += len(y)
    return correct / max(n, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/protocol_v1.yaml")
    ap.add_argument("--root", default="data")
    ap.add_argument("--adapter-path", required=True, help="Path to adapter.pt (e.g. from Drive)")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--out", default="results/final_eval")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_y = yaml.safe_load(open(a.config))
    tf = build_transform(cfg_y["preprocessing"])

    print("Loading model...")
    clip, tok, _ = load_clip(cfg_y["base_model"]["hf_id"], cfg_y["base_model"].get("revision"), device)
    model = ClipLoRA(clip, rank=a.rank, alpha=a.alpha or 2*a.rank).to(device)
    
    print(f"Loading adapter from {a.adapter_path}...")
    st = torch.load(a.adapter_path, map_location=device)
    model.load_adapter_state(st)
    model.eval()

    # Map probe names to their templates
    prompt_map = {
        "cifar100": CIFAR100_TEMPLATES,
        "food101": FOOD101_TEMPLATES
    }

    results = {}
    for name in ZS_TEST:
        print(f"Evaluating {name}...")
        # unlock_test=True is required for ZS-test probes
        p = load_probe(name, a.root, tf, max_images=0, unlock_test=True) 
        
        templates = prompt_map[name]
        ct = class_embeddings(model, tok, p["classnames"], lambda c: [t.format(c) for t in templates], device)
        
        loader = DataLoader(p["dataset"], batch_size=256, num_workers=2)
        acc = evaluate(model, loader, ct, device)
        results[name] = acc
        print(f"{name}: {acc:.4f}")

    os.makedirs(a.out, exist_ok=True)
    import json
    with open(os.path.join(a.out, "final_test_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {a.out}/final_test_results.json")

if __name__ == "__main__":
    main()