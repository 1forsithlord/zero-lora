# Novelty matrix, claims ledger and novelty gate (Phase 1)

Sources: `docs/literature_notes.md`. "?" = not checked or only abstract-level. A search that finds nothing does **not** prove absence; claims below are written so they survive an overlap being found later.

## 1. Comparison matrix

| Dimension | ZSCL | GNSP | NuSA-CL | ZAF | MoE-Adapters | Niss et al. | **Ours (planned)** |
|---|---|---|---|---|---|---|---|
| Backbone | CLIP | CLIP ViT-B/16 | CLIP ViT-B/16 (MTIL) | BLIP | CLIP | CLIP + 3 others | CLIP ViT-B/16 |
| What is trained | full FT | FFN of image encoder (projected full-rank) | null-space low-rank, q/k/v/o, both encoders | EMA-LoRA r=16 | LoRA experts + routers | full FT, PEFT incl. LoRA | LoRA (q,v then ablate), text frozen |
| Preservation mechanism | similarity-distribution distillation (softmax CE) + weight avg | CD (row/col KL on similarity matrices) + MAP + null-space | null-space constraint from weights | L1 on output probs vs previous step | routing to frozen CLIP | none (prediction only) | **raw Frobenius similarity matching** vs distributional distillation |
| Reference data | ImageNet imgs + CC texts | 1k ImageNet imgs | none | wild VG/VAW set | n/a (DDAS) | none | CC3M subset, leakage-audited |
| Normalization before matching | softmax | softmax | n/a | softmax probs | n/a | n/a | **none (raw S = tau cos)** |
| Drift analysis | no | modality-gap trajectories (figures, correlational) | spectral only | no | ? | pre-FT predictor IIMM, final ckpts | **checkpoint-level, lead-lag, vs weight-norm control** |
| Benchmark | MTIL | MTIL | MTIL, CIFAR-100 CIL | VG/VAW | MTIL | 9 datasets | DTD, EuroSAT, +2-4 tasks (own protocol) |
| Merging | no | no | merge into backbone per task | no | no | no | optional RQ3 (fixed vs scalar) |

## 2. Claims ledger (every claim tied to evidence; nothing is a result yet)

| ID | Claim (draft wording) | Type | Evidence needed | Status |
|---|---|---|---|---|
| C1 | Under matched backbone, data, LoRA setup and budget, raw similarity-geometry regularization changes the task-accuracy / zero-shot trade-off relative to plain LoRA and to distributional distillation. | Empirical, RQ1 | Phase 4 table, multi-seed, test pool evaluated once | OPEN |
| C2 | Direction of the change (better / equal / worse) is reported as found. | Empirical | same | OPEN |
| C3 | Drift at step t carries information about zero-shot degradation at t+k beyond weight-update norm. | Empirical, RQ2 | Phase 5, within-run and across-run analysis | OPEN (null result acceptable) |
| C4 | Scalar learned merge vs fixed merge under identical fitting data. | Empirical, RQ3 | Phase 7 | OPEN, optional |
| N1 | "Raw similarity matching is a new loss." | Novelty | would need exhaustive search | **DO NOT CLAIM** (logit/KD matching is classical) |
| N2 | "First to regularize LoRA with a reference set." | Novelty | ZAF (BLIP, LoRA) already exists | **DO NOT CLAIM** |
| N3 | "Drift predicts forgetting." | Novelty | Niss et al. and GNSP are adjacent | **DO NOT CLAIM as new in general**; claim only the specific controlled test in C3 |

## 3. Novelty gate (roadmap requirement: assess before any claim)

1. **Is the loss new?** Probably not in any strong sense. Matching teacher and student similarity values on a reference set is covered in spirit by ZSCL/GNSP (softmax-normalized) and by classical logit distillation (raw). What I did not find is a *controlled head-to-head of raw vs softmax-normalized similarity matching on LoRA*. Treat that as the contribution: **a controlled comparison**, not a method.
2. **Is the setting new?** LoRA + reference-set regularizer exists (ZAF, BLIP). LoRA + CLIP + raw matching on MTIL-style tasks: not found, not proven absent.
3. **Is the diagnostic new?** Nearby but different (Niss: pre-hoc across tasks; GNSP: figure-level gap trajectories). The lead-lag test against a weight-norm baseline is the distinct piece. It is also the piece that could simply return a null result.

**Gate verdict (draft, for Yash to confirm):** proceed, with claims narrowed to C1-C3 as controlled empirical questions. The distributional baseline must be **both** a ZSCL-style softmax-CE variant and a GNSP-CD-style row/column KL variant (they differ), implemented from the papers/official code, and described as "re-implementation" until checked against code.

## 4. Design consequences carried into the protocol
1. Distillation baselines: ZSCL-style (softmax CE, both directions) and GNSP-CD-style (row/col KL, tau-scaled). Raw loss: Frobenius, detached teacher.
2. **Scale issue to settle in Phase 3:** S = tau·cos with tau about 100 makes the raw loss scale with tau^2; softmax-based losses do not. Normalize by (N·M) as in the roadmap and report lambda on a log grid chosen on DEV only; compare methods at matched *task accuracy* or on the full Pareto curve, not at one arbitrary lambda.
3. **One more control worth adding (cheap):** the same raw loss on *logits of the original contrastive batch* (in-batch pairs) vs a separate reference batch, to separate "raw vs softmax" from "reference data vs none".
4. Reporting: absolute task accuracy, absolute zero-shot accuracy, delta, retention ratio only alongside the absolutes.
