"""LoRA on CLIP's vision q,v projections; text encoder frozen; teacher = the same network with LoRA off.

Design notes
- Own minimal LoRA (not peft) so that (a) the teacher is *exactly* the initial CLIP (adapter switched off,
  base weights never touched), (b) behaviour is bit-reproducible, (c) merging later operates on dW = (alpha/r) B A.
- Teacher without a second model copy: `with model.lora_disabled(): ...` (saves ~600 MB on a T4).
- `logit_scale` is CLIP's exp(logit_scale) (= 100). This is what the roadmap calls "tau" in S = tau * cos.
  [NOTATION CONFLICT, see docs/phase2_notes.md: GNSP writes Z = S/tau with tau = 0.01; here the scale is 100.]
"""
from __future__ import annotations

import contextlib
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_tensor(f):
    """transformers versions differ in what get_*_features returns; accept a tensor or an output object."""
    if torch.is_tensor(f):
        return f
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        v = getattr(f, attr, None)
        if v is not None:
            return v
    raise TypeError(f"cannot extract features from {type(f)}")


class LoRALinear(nn.Module):
    """y = W x + b + (alpha/r) * B A x ;  A ~ kaiming-uniform(a=sqrt(5)) (as in the LoRA reference), B = 0."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        assert r > 0
        self.base, self.r, self.alpha, self.scaling = base, r, float(alpha), float(alpha) / r
        w = base.weight
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=w.device, dtype=w.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=w.device, dtype=w.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.enabled = True

    def forward(self, x):
        out = self.base(x)
        if not self.enabled:
            return out
        return out + F.linear(F.linear(self.drop(x), self.lora_A), self.lora_B) * self.scaling

    def delta_weight(self):
        return self.scaling * (self.lora_B @ self.lora_A)


class ClipLoRA(nn.Module):
    def __init__(self, clip, rank=4, alpha=None, dropout=0.0, targets=("q_proj", "v_proj"), logit_scale=None):
        super().__init__()
        self.clip = clip
        self.clip.eval()
        for p in self.clip.parameters():
            p.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha if alpha is not None else 2 * rank)
        self.targets = tuple(targets)
        self.lora_modules = []
        if self.rank > 0:
            for layer in clip.vision_model.encoder.layers:
                for t in self.targets:
                    base = getattr(layer.self_attn, t)
                    assert isinstance(base, nn.Linear), f"{t} is {type(base)}"
                    w = LoRALinear(base, self.rank, self.alpha, dropout)
                    setattr(layer.self_attn, t, w)
                    self.lora_modules.append(w)
        self._logit_scale = float(logit_scale) if logit_scale is not None else float(clip.logit_scale.exp().item())
        self.register_buffer("class_text", torch.zeros(0), persistent=False)

    # ---- bookkeeping -------------------------------------------------------------------------------
    @property
    def logit_scale(self):
        return self._logit_scale

    def train(self, mode=True):
        super().train(mode)
        self.clip.eval()                      # CLIP stays in eval(); only LoRA dropout (if any) follows `mode`
        for m in self.lora_modules:
            m.train(mode)
        return self

    def lora_parameters(self):
        return [p for n, p in self.named_parameters() if "lora_" in n]

    def num_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @contextlib.contextmanager
    def lora_disabled(self):
        prev = [m.enabled for m in self.lora_modules]
        for m in self.lora_modules:
            m.enabled = False
        try:
            yield
        finally:
            for m, p in zip(self.lora_modules, prev):
                m.enabled = p

    def assert_frozen(self):
        """Raise if any non-LoRA parameter is trainable, or (rank>0) any LoRA parameter is not."""
        bad = [n for n, p in self.named_parameters() if "lora_" not in n and p.requires_grad]
        if bad:
            raise RuntimeError(f"non-LoRA parameters are trainable: {bad[:5]} ...")
        if self.rank > 0:
            off = [n for n, p in self.named_parameters() if "lora_" in n and not p.requires_grad]
            if off:
                raise RuntimeError(f"LoRA parameters are frozen: {off[:5]} ...")
        if self.clip.logit_scale.requires_grad:
            raise RuntimeError("logit_scale must be frozen")

    # ---- encoders ----------------------------------------------------------------------------------
    def encode_image(self, pixel_values):
        return F.normalize(_as_tensor(self.clip.get_image_features(pixel_values=pixel_values)), dim=-1)

    @torch.no_grad()
    def encode_text(self, input_ids, attention_mask=None):
        f = self.clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        return F.normalize(_as_tensor(f), dim=-1)

    def set_class_text(self, emb):
        self.class_text = F.normalize(emb.detach(), dim=-1)

    def logits(self, img_emb, class_text=None):
        ct = self.class_text if class_text is None else class_text
        return self._logit_scale * img_emb @ ct.T

    # ---- API requested in the Phase 2 spec ---------------------------------------------------------
    def forward_task(self, batch):
        """batch: (pixel_values, labels) or pixel_values -> task logits scale*cos(img, class_text)."""
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        return self.logits(self.encode_image(x))

    def forward_reference(self, batch):
        """batch: dict(pixel_values, input_ids, attention_mask) -> (student image_emb [grad], text_emb [no grad])."""
        img = self.encode_image(batch["pixel_values"])
        txt = self.encode_text(batch["input_ids"], batch.get("attention_mask"))
        return img, txt

    @torch.no_grad()
    def forward_teacher(self, batch):
        """Initial-CLIP image embedding for the same reference batch (adapter off, no grad)."""
        with self.lora_disabled():
            return self.encode_image(batch["pixel_values"])

    # ---- persistence -------------------------------------------------------------------------------
    def adapter_state(self):
        return {"rank": self.rank, "alpha": self.alpha, "targets": self.targets,
                "weights": {n: p.detach().cpu() for n, p in self.named_parameters() if "lora_" in n}}

    def load_adapter_state(self, st):
        own = dict(self.named_parameters())
        for n, v in st["weights"].items():
            own[n].data.copy_(v)


def resolve_model_commit(hf_id, revision=None):
    try:
        from huggingface_hub import snapshot_download
        return os.path.basename(snapshot_download(hf_id, revision=revision, allow_patterns=["config.json"]))
    except Exception:
        return None


def load_clip(hf_id, revision=None, device="cpu", attn_impl="eager"):
    """Load HF CLIP + tokenizer. attn_impl='eager' avoids nondeterministic fused-attention backward kernels."""
    from transformers import CLIPModel, CLIPTokenizer
    model = CLIPModel.from_pretrained(hf_id, revision=revision, attn_implementation=attn_impl).to(device).eval()
    tok = CLIPTokenizer.from_pretrained(hf_id, revision=revision)
    return model, tok, resolve_model_commit(hf_id, revision)