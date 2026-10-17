"""Preservation losses. All three take the SAME inputs: logit matrices S (N images x M texts) with
S = logit_scale * cos(img, txt), student with grad and teacher detached. They differ only in how S is
normalized before the comparison (none / softmax-CE / softmax-KL), which is the controlled variable of RQ1.
"""
import torch.nn.functional as F

from .gnsp_cd import gnsp_cd
from .raw_frobenius import raw_frobenius
from .zscl_ce import teacher_entropy, zscl_ce

LOSSES = {"raw": raw_frobenius, "zscl": zscl_ce, "gnsp": gnsp_cd}


def similarity_logits(img_emb, txt_emb, logit_scale):
    """S = logit_scale * cos(img, txt) for L2-normalized embeddings -> [N, M]."""
    return logit_scale * (F.normalize(img_emb, dim=-1) @ F.normalize(txt_emb, dim=-1).T)


def get_loss(name):
    if name not in LOSSES:
        raise KeyError(f"unknown loss {name!r}; choose from {sorted(LOSSES)}")
    return LOSSES[name]


__all__ = ["LOSSES", "get_loss", "similarity_logits", "raw_frobenius", "zscl_ce", "gnsp_cd", "teacher_entropy"]