"""Loader for the frozen reference pool built by scripts/build_reference_pool.py."""
import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset

from .poolutils import compute_pool_hash, read_captions


class ReferencePool(Dataset):
    def __init__(self, root, transform, verify=True):
        self.root, self.transform = root, transform
        self.records = read_captions(os.path.join(root, "captions.jsonl"))
        hp = os.path.join(root, "pool_hash.json")
        self.pool_hash = compute_pool_hash(self.records)
        if verify and os.path.exists(hp):
            want = json.load(open(hp))["pool_hash_sha256"]
            if want != self.pool_hash:
                raise RuntimeError(f"reference pool changed since it was frozen: {self.pool_hash} != {want}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        img = Image.open(os.path.join(self.root, r["path"])).convert("RGB")
        return self.transform(img), r["caption"]


def make_collate(tokenizer, max_length=77):
    def collate(batch):
        imgs, caps = zip(*batch)
        t = tokenizer(list(caps), padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
        return {"pixel_values": torch.stack(imgs), "input_ids": t["input_ids"], "attention_mask": t["attention_mask"]}
    return collate