"""Zero-shot probe datasets (retention is measured here) and class-text embeddings.

ZS-dev  = caltech101, oxford_pets   -> usable for lambda selection and logging during training.
ZS-test = cifar100, food101         -> LOCKED: load_probe() refuses them unless unlock_test=True, and
                                       train.py never passes it. They are touched once, by the final script.

Prompt sources: Caltech101 (34 templates) and OxfordPets (1 template) are the openai/CLIP lists, verified
2026-10-16 against data/prompts.md. Class names: Oxford Pets = torchvision names as-is; Caltech101 =
torchvision category names, lower-cased with '_' -> ' ', plus CLIP's four renamings. This is NOT CLIP's
exact cleaned Caltech list (the full list was not retrievable); it is dev-only and logged in
docs/protocol_changelog.md. Probes are evaluated on a fixed seeded subsample (hash recorded).
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from .splits import index_hash

ZS_DEV = ("caltech101", "oxford_pets")
ZS_TEST = ("cifar100", "food101")

CALTECH_TEMPLATES = [
    "a photo of a {}.", "a painting of a {}.", "a plastic {}.", "a sculpture of a {}.", "a sketch of a {}.",
    "a tattoo of a {}.", "a toy {}.", "a rendition of a {}.", "a embroidered {}.", "a cartoon {}.",
    "a {} in a video game.", "a plushie {}.", "a origami {}.", "art of a {}.", "graffiti of a {}.",
    "a drawing of a {}.", "a doodle of a {}.", "a photo of the {}.", "a painting of the {}.", "the plastic {}.",
    "a sculpture of the {}.", "a sketch of the {}.", "a tattoo of the {}.", "the toy {}.",
    "a rendition of the {}.", "the embroidered {}.", "the cartoon {}.", "the {} in a video game.",
    "the plushie {}.", "the origami {}.", "art of the {}.", "graffiti of the {}.", "a drawing of the {}.",
    "a doodle of the {}.",
]
PETS_TEMPLATES = ["a photo of a {}, a type of pet."]
_CALTECH_RENAME = {"Faces": "off-center face", "Faces_easy": "centered face",
                   "Leopards": "leopard", "Motorbikes": "motorbike"}


class _Wrap(Dataset):
    def __init__(self, base, indices, label_map=None):
        self.base, self.indices, self.label_map = base, list(indices), label_map

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, y = self.base[self.indices[i]]
        if self.label_map is not None:
            y = self.label_map[y]
        return img, y


def load_probe(name, root, transform, max_images=2000, seed=1234, unlock_test=False):
    if name in ZS_TEST and not unlock_test:
        raise PermissionError(f"{name} is a ZS-TEST probe: locked until the final evaluation (leakage checklist A).")
    import torchvision.datasets as D
    if name == "caltech101":
        ds = D.Caltech101(root, target_type="category", transform=transform, download=True)
        cats = [c for c in ds.categories if c != "BACKGROUND_Google"]
        old2new = {ds.categories.index(c): i for i, c in enumerate(cats)}
        cand = [i for i, y in enumerate(ds.y) if y in old2new]
        classnames = [_CALTECH_RENAME.get(c, c.replace("_", " ").lower()) for c in cats]
        label_map, templates = old2new, CALTECH_TEMPLATES
    elif name == "oxford_pets":
        ds = D.OxfordIIITPet(root, split="test", target_types="category", transform=transform, download=True)
        cand, classnames, label_map, templates = list(range(len(ds))), list(ds.classes), None, PETS_TEMPLATES
    else:
        raise KeyError(f"unknown probe {name}")
    if max_images and max_images < len(cand):
        sel = np.sort(np.random.RandomState(seed).choice(len(cand), max_images, replace=False))
        cand = [cand[i] for i in sel]
    return {"name": name, "dataset": _Wrap(ds, cand, label_map), "classnames": classnames,
            "templates": templates, "n": len(cand), "hash": index_hash(cand)}


@torch.no_grad()
def class_embeddings(model, tokenizer, classnames, prompts_fn, device):
    """Mean of L2-normalized template embeddings per class, renormalized -> [C, D]. `model` is a ClipLoRA."""
    out = []
    for c in classnames:
        ids = tokenizer(prompts_fn(c), padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        e = model.encode_text(ids["input_ids"].to(device), ids["attention_mask"].to(device))
        e = e.mean(0)
        out.append(e / e.norm())
    return torch.stack(out)