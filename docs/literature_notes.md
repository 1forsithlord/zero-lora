# Literature notes (Phase 1, Day 1-2)

Status legend: **[READ]** = method section read this session via fetched full text; **[ABS]** = abstract/search snippet only; **[VERIFY]** = from memory or secondary source, must be checked against the paper before it is cited or used in a claim.

Nothing here is a result of ours. All numbers belong to the cited papers.

---

## 1. ZSCL — Zero-shot Continual Learning of VLMs (ICCV 2023, arXiv 2303.06628) [READ]
- **Preservation mechanism:** feature-space distillation of *relative similarity distributions*. For an image, p = softmax over its similarities to the texts; the student's distribution is matched to the teacher's. Image direction written as cross-entropy CE(p, p̄) (not written as KL); a text-direction term penalizes changes in a text's distances to a batch of images (no explicit equation in the paper).
- **Teacher:** the *initial* pretrained CLIP (not the previous-task model).
- **Reference data:** images sampled from ImageNet (100k for MTIL); texts from Conceptual Captions for MTIL. Paper's Table 1 caption and Sec. 5 disagree on the text source, so check the official code before calling anything a faithful reproduction.
- **Total loss:** L = L_ce + λ(L_lwf_img + L_lwf_txt), λ = 1.
- **Parameter-space part:** running weight average, every I = 100 iterations: θ̂_t = θ_t/(t+1) + t·θ̂_{t-1}/(t+1).
- **Setting:** full fine-tuning (not LoRA). MTIL metrics: Transfer (upper-right triangle), Avg, Last. Order-I: Aircraft, Caltech101, CIFAR100, DTD, EuroSAT, Flowers, Food, MNIST, OxfordPet, StanfordCars, SUN397.
- **Relevance:** this is the "distributional distillation" baseline family. It matches *softmaxed* similarities, so it is invariant to a per-row additive shift of the logits. Our raw-geometry loss is not.

## 2. GNSP — gradient null-space projection + modality alignment (arXiv 2507.19839) [READ, excerpt]
- **Contrastive Distillation (CD):** frozen initial CLIP teacher; batch cosine-similarity matrices S⁰, Sᵗ divided by τ; row-wise KL (image-to-text) + column-wise KL (text-to-image); λ = 1; 1,000 random ImageNet reference images (their ablation: 1k better than 10k).
- **MAP loss:** symmetric in-batch contrastive loss on reference pairs (positives = diagonal). It is computed against the ground-truth pairing, not against the teacher; β = 0.75.
- **Null-space projection:** applied to the 12 FFN layers of the image encoder only; ΔW = P·G (full-rank FFN update, **not LoRA**); P from SVD of accumulated activation Gram matrices with energy threshold ρ = 0.15.
- **Drift/gap analysis:** mean paired image-text cosine similarity ("modality gap" in their usage) tracked per task, shown in figures; authors state the gap is *not a direct cause* of degradation but sharp variations are an *indicator correlated* with zero-shot loss. One backbone (ViT-B/16). Evidence is correlational.
- **Relevance (high risk for novelty):** CD is the closest published relative of our "similarity-matrix distillation" baseline. Treat GNSP-CD as the canonical distributional baseline. Their drift-as-indicator analysis partly overlaps RQ2 but is figure-level, not a statistical test against a weight-norm control.

## 3. NuSA-CL — Null-Space Adaptation for continual learning (arXiv 2510.21175) [READ, excerpt]
- Null space from SVD of each *frozen weight matrix* before each task; no task data/features/gradients. Smallest top-k subspace covering fraction ρ of spectral energy (ρ = 0.99 for 5-shot, 0.95 full-shot); r_max = 128.
- Update ΔW = U_n M V_nᵀ, only the r×r matrix M trained (init 0); merged into the backbone after each task (no extra modules).
- Adapters on **q, k, v, o of every attention block, in both vision and text encoders**; vision-only adaptation worse than both (their ablation).
- Benchmarks: MTIL (full-shot and 5-shot), class-incremental CIFAR100. Baselines include ZSCL, MoE-Adapters, DIKI, InfLoRA, LoRA, MiLoRA.
- **No explicit drift metric**; spectral analysis (effective rank, null ratio) only. No code link in the fetched text.
- **Relevance:** a *constraint-based* alternative to our *regularization-based* approach. Not a direct competitor for RQ1 but must be cited and, if compute allows, used as a reference point. Note it adapts the text encoder; our primary experiment freezes it (a stated difference, not an advantage).

## 4. ZAF — Stabilizing Zero-Shot Prediction (NeurIPS 2024) [READ, excerpt]
- Adds L_ZS: L1 difference between current and *previous-step* (EMA) model's output probabilities on an unlabeled, unpaired "wild" set (about 12k images from VG/VAW, nouns + ChatGPT-4 concept words as texts). Replay-free.
- **EMA-LoRA** (rank 16, α = 0.85) on **BLIP**; benchmarks are VG/VAW sequences, **not MTIL**. Compared against ZSCL, MoE-Adapters, LwF, LoRA, ConStruct-VL.
- **Relevance:** shows LoRA + output-matching on an unlabeled reference set is already published, for BLIP. So "LoRA + reference-set regularizer" is not novel by itself.

## 5. Niss et al. — Zero-Shot Embeddings Inform Learning and Forgetting (arXiv 2407.15731) [READ, excerpt]
- Proposes IIMM (Inter-Intra Modal Measure) from *zero-shot embeddings before fine-tuning*; predicts in-domain gain (Spearman about 0.88-0.95 across 4 backbones) and, except for CLIP, forgetting on off-target tasks.
- Evaluated full FT on CLIP/SigLIP/CoCa/EVA-02-CLIP and PEFT (incl. LoRA) on CLIP ViT-B/32; **final checkpoints only**, IIMM computed once pre-training. States forgetting is not strongly linked to IIMM for CLIP.
- **Relevance:** it is a *pre-hoc predictor across tasks*. Our RQ2 is a *during-training* drift signal tested against a weight-norm control with lead-lag analysis. Different question; must be cited and contrasted. Their CLIP-specific negative finding is a warning for RQ2.

## 6. MoE-Adapters (CVPR 2024, arXiv 2403.11549, Yu et al.) [ABS verified 2026-10-16; body not read]
- Abstract confirms: CLIP is expanded with Mixture-of-Experts adapters as tasks arrive; a Distribution Discriminative Auto-Selector (DDAS) sends in-distribution inputs to the MoE adapters and out-of-distribution inputs to the original CLIP to protect zero-shot recognition; parameter count grows with tasks; reports about 60% less parameter training burden.
- **Not confirmed from the abstract:** whether the experts are LoRA, the router internals, and per-task routing details. Earlier descriptions of these in our plan are unverified. Do not write them into the paper until Sec. 3 is read.
- **Team action:** read Sec. 3 and replace this block with 5 exact lines. Still unverified: the external review's "MoE-Adapters++" TPAMI claim.

## 7. LoRA composition / merging (Day 2) [ABS only: titles from search, no IDs recorded]
- A rank-1 expert pool style approach, "Merge-before-Forget", and "Aligned Model Merging": abstracts only. None read in full. Add IDs/venues and 5-line notes after reading.
- Standard merging references the team must read in full before RQ3: Task Arithmetic (Ilharco et al.), TIES-Merging (Yadav et al.), DARE (Yu et al.), WiSE-FT (Wortsman et al.) [VERIFY years/venues].
- Rule carried from the roadmap: merge on the products ΔW = (s/r)BA, never on A and B separately.

## 8. Other items found [ABS]
- A 2026 preprint treating attention/representation drift as a *descriptive, non-causal* diagnostic of CLIP retention (title/ID not recorded) [VERIFY]. Read before writing RQ2 related work.

---

## Open reading list for the team (do before Gate 1)
| Who | Item | Output |
|---|---|---|
| Yash Pratap Singh | ZSCL official code (check text-source and loss details) | notes in section 1 |
| Shreyash Joshi | MoE-Adapters full paper Sec. 3 | replace section 6 |
| Yash Raj | the 2026 attention-drift preprint + 3 LoRA-merging abstracts to full reads | replace sections 7-8 |
