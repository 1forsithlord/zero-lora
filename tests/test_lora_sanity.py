"""Phase 2C sanity tests. Run offline on a tiny random CLIP (no downloads, CPU):  python tests/test_lora_sanity.py
Findings that deviate from the Phase 2 request are asserted explicitly and explained in docs/phase2_notes.md:
  * zscl_ce(student == teacher) equals the teacher ENTROPY, not 0 (cross-entropy has a floor); excess CE is 0.
  * zscl_ce and gnsp_cd have identical gradients (CE = KL + constant), so they are one optimizer-level baseline.
"""
import copy
import math
import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import CLIPConfig, CLIPModel, CLIPTextConfig, CLIPVisionConfig

from zerolora.lora import ClipLoRA, _as_tensor
from zerolora.losses import gnsp_cd, raw_frobenius, similarity_logits, teacher_entropy, zscl_ce, LOSSES
from zerolora.train import TrainConfig, run_training

D = 16


def tiny_clip(seed=0):
    torch.manual_seed(seed)
    tc = CLIPTextConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                        num_attention_heads=2, max_position_embeddings=16, projection_dim=D,
                        bos_token_id=0, eos_token_id=2)
    vc = CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                          image_size=32, patch_size=16, projection_dim=D)
    cfg = CLIPConfig(text_config=tc.to_dict(), vision_config=vc.to_dict(), projection_dim=D,
                     logit_scale_init_value=math.log(100.0))
    return CLIPModel(cfg).eval()


class _Ref(Dataset):
    def __init__(self, n=48, seed=5):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, 3, 32, 32, generator=g)
        self.ids = torch.randint(3, 60, (n, 16), generator=g)

    def __len__(self): return len(self.x)

    def __getitem__(self, i):
        return {"pixel_values": self.x[i], "input_ids": self.ids[i], "attention_mask": torch.ones(16, dtype=torch.long)}


def _data(seed=0, n=64, classes=5):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, 32, 32, generator=g); y = torch.randint(0, classes, (n,), generator=g)
    train = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True, generator=torch.Generator().manual_seed(seed))
    dev = DataLoader(TensorDataset(x[:32], y[:32]), batch_size=16)
    probe = DataLoader(TensorDataset(torch.randn(24, 3, 32, 32, generator=g), torch.randint(0, 3, (24,), generator=g)), batch_size=12)
    ref = DataLoader(_Ref(), batch_size=12, shuffle=True, drop_last=True, generator=torch.Generator().manual_seed(seed + 10_000))
    ct_task = F.normalize(torch.randn(classes, D, generator=g), dim=-1)
    ct_probe = F.normalize(torch.randn(3, D, generator=g), dim=-1)
    return train, dev, {"probeA": (probe, ct_probe)}, ref, ct_task


def _run(loss="raw", lam=0.0, force_ref=False, use_ref=True, seed=0, epochs=2, rank=4):
    clip = tiny_clip(seed)
    model = ClipLoRA(clip, rank=rank, alpha=8)
    train, dev, probes, ref, ct = _data(seed)
    model.set_class_text(ct)
    cfg = TrainConfig(loss=loss, lam=lam, rank=rank, alpha=8, epochs=epochs, lr=1e-2, seed=seed, batch_size=16,
                      force_ref=force_ref, log_every=1)
    with tempfile.TemporaryDirectory() as d:
        s = run_training(model, train, dev, probes, ref if use_ref else None, cfg, d)
        files = set(os.listdir(d))
    return {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n}, s, files


def _same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


# ---------------------------------------------------------------------------------------------- tests
def test_init_and_rank0_identical_to_base():
    base = tiny_clip(0)
    pristine = copy.deepcopy(base)
    x = torch.randn(4, 3, 32, 32)
    for rank in (0, 4):
        m = ClipLoRA(copy.deepcopy(pristine), rank=rank)
        ref = F.normalize(_as_tensor(pristine.get_image_features(pixel_values=x)), dim=-1)
        assert torch.equal(m.encode_image(x), ref), f"rank={rank}: output differs from base CLIP at init"
        if rank == 0:
            assert m.lora_parameters() == [] and m.num_trainable() == 0
        else:
            assert len(m.lora_parameters()) == 2 * 2 * 2  # 2 layers x (q,v) x (A,B)
        m.assert_frozen()


def test_lambda0_is_bitwise_plain_lora():
    plain, s0, files = _run(use_ref=False, lam=0.0)
    plain2, _, _ = _run(use_ref=False, lam=0.0)
    assert _same(plain, plain2), "plain LoRA is not reproducible under a fixed seed"
    assert not s0["used_reference_path"] and {"summary.json", "metrics.jsonl", "adapter.pt"} <= files
    assert any(p.abs().sum() > 0 for k, p in plain.items() if "lora_B" in k), "training did not move LoRA B"
    for loss in LOSSES:
        forced, s1, _ = _run(loss=loss, lam=0.0, force_ref=True, use_ref=True)
        assert s1["used_reference_path"]
        assert _same(plain, forced), f"lambda=0 with the reference path active ({loss}) != plain LoRA"
        unforced, s2, _ = _run(loss=loss, lam=0.0, force_ref=False, use_ref=True)
        assert not s2["used_reference_path"] and _same(plain, unforced)
    for loss in LOSSES:   # and a non-zero lambda must actually change the result
        withreg, _, _ = _run(loss=loss, lam=1e-3, use_ref=True)
        assert not _same(plain, withreg), f"{loss}: lambda>0 had no effect"


def test_no_gradient_into_text_encoder_or_teacher():
    for loss in LOSSES:
        clip = tiny_clip(0)
        pristine = copy.deepcopy(clip)
        m = ClipLoRA(clip, rank=4, alpha=8)
        for p in m.lora_parameters():                       # make student != teacher
            p.data.add_(0.05 * torch.randn_like(p))
        batch = next(iter(_data(0)[3]))
        img_s, txt = m.forward_reference(batch)
        img_t = m.forward_teacher(batch)
        assert not img_t.requires_grad and not txt.requires_grad, "teacher/text path carries gradients"
        S_s = similarity_logits(img_s, txt, m.logit_scale); S_t = similarity_logits(img_t, txt, m.logit_scale)
        loss_v = LOSSES[loss](S_s, S_t)
        loss_v.backward()
        for n, p in m.named_parameters():
            if "lora_" not in n:
                assert p.grad is None, f"{loss}: gradient reached frozen parameter {n}"
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.lora_parameters())
        # teacher == the ORIGINAL clip even after the adapter moved
        x = batch["pixel_values"]
        assert torch.equal(m.forward_teacher(batch), F.normalize(_as_tensor(pristine.get_image_features(pixel_values=x)), dim=-1))
        m.assert_frozen()
    bad = ClipLoRA(tiny_clip(0), rank=2)
    next(bad.clip.text_model.parameters()).requires_grad_(True)
    try:
        bad.assert_frozen(); raise AssertionError("assert_frozen should have raised")
    except RuntimeError:
        pass


def test_losses_at_student_equals_teacher():
    torch.manual_seed(0)
    for shape in ((8, 8), (6, 9)):
        S = 100.0 * torch.randn(*shape).clamp(-1, 1)
        assert raw_frobenius(S.clone(), S).item() == 0.0
        assert abs(gnsp_cd(S.clone(), S).item()) < 1e-5
        H = teacher_entropy(S)
        assert torch.allclose(zscl_ce(S.clone(), S), H, atol=1e-5), "CE at student==teacher must equal teacher entropy"
        assert abs((zscl_ce(S.clone(), S) - H).item()) < 1e-5     # excess CE is 0
    # through the real wrapper: B=0 -> student embedding == teacher embedding exactly
    m = ClipLoRA(tiny_clip(0), rank=4)
    batch = next(iter(_data(0)[3]))
    img_s, txt = m.forward_reference(batch); img_t = m.forward_teacher(batch)
    S_s = similarity_logits(img_s, txt, m.logit_scale); S_t = similarity_logits(img_t, txt, m.logit_scale)
    assert raw_frobenius(S_s, S_t).item() == 0.0 and abs(gnsp_cd(S_s, S_t).item()) < 1e-6


def test_zscl_and_gnsp_have_identical_gradients_raw_differs():
    torch.manual_seed(1)
    St = 30 * torch.randn(8, 8)
    Ss = (St + 3 * torch.randn(8, 8)).requires_grad_(True)
    g = {}
    for name, fn in (("zscl", zscl_ce), ("gnsp", gnsp_cd), ("raw", raw_frobenius)):
        Ss.grad = None; fn(Ss, St).backward(); g[name] = Ss.grad.clone()
    assert torch.allclose(g["zscl"], g["gnsp"], atol=1e-6), "CE and KL should differ only by a constant"
    assert not torch.allclose(g["zscl"], g["raw"], atol=1e-3)
    v = zscl_ce(Ss.detach(), St) - gnsp_cd(Ss.detach(), St)
    assert torch.allclose(v, teacher_entropy(St), atol=1e-5)          # value gap == teacher entropy
    lit = zscl_ce(Ss.detach(), St, orientation="student_target")
    assert not torch.allclose(lit, zscl_ce(Ss.detach(), St))          # literal orientation is a different objective


def test_loss_scales_with_lambda_on_log_grid():
    m = ClipLoRA(tiny_clip(0), rank=4, alpha=8)
    for p in m.lora_parameters():
        p.data.add_(0.05 * torch.randn_like(p))
    batch = next(iter(_data(0)[3]))
    grid = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
    for name, fn in LOSSES.items():
        norms, vals = [], []
        for lam in grid:
            m.zero_grad(set_to_none=True)
            img_s, txt = m.forward_reference(batch); img_t = m.forward_teacher(batch)
            l = fn(similarity_logits(img_s, txt, m.logit_scale), similarity_logits(img_t, txt, m.logit_scale))
            assert torch.isfinite(l) and l.item() > 0, f"{name}: loss must be finite and positive off-teacher"
            (lam * l).backward()
            norms.append(torch.sqrt(sum((p.grad ** 2).sum() for p in m.lora_parameters())).item()); vals.append(lam * l.item())
        for i in range(1, len(grid)):
            assert math.isclose(vals[i] / vals[i - 1], grid[i] / grid[i - 1], rel_tol=1e-4)
            assert math.isclose(norms[i] / norms[i - 1], grid[i] / grid[i - 1], rel_tol=1e-3), (name, norms)
    # raw: exactly mean squared difference, quadratic in the perturbation
    St = 20 * torch.randn(5, 7); d = torch.randn(5, 7)
    assert math.isclose(raw_frobenius(St + d, St).item(), (d ** 2).mean().item(), rel_tol=1e-6)
    assert math.isclose(raw_frobenius(St + 2 * d, St).item() / raw_frobenius(St + d, St).item(), 4.0, rel_tol=1e-5)


def test_trainer_outputs_and_frozen_eval_is_epoch0():
    params, s, files = _run(loss="gnsp", lam=0.1, use_ref=True, epochs=2)
    assert s["frozen"]["dev_task_acc"] is not None and set(s["frozen"]["zs"]) == {"probeA"}
    assert s["steps"] == 2 * 4 and s["trainable_params"] == sum(p.numel() for p in params.values())
    assert s["used_reference_path"] and s["retention_ratio_zs_mean"] is not None
    assert {"summary.json", "metrics.jsonl", "adapter.pt"} <= files


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
    print("ALL LoRA SANITY TESTS PASSED")