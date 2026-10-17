# Phase 2 notes: assumptions, flags, findings, what is and is not tested

## What was tested (offline, CPU, tiny random CLIP, transformers 5.20 / torch 2.14)
`tests/test_lora_sanity.py` (7 tests), `tests/test_refpool.py` (5), `tests/test_splits.py` (4), all passing:
- rank 0 and rank>0-at-init outputs are bit-identical to the base CLIP.
- lambda = 0 reproduces plain LoRA **bit-for-bit** (all LoRA parameters `torch.equal`) for all three losses, both with the reference path skipped and with it forced-on and multiplied by 0.0. Lambda > 0 changes the result. Plain LoRA is reproducible run-to-run.
- no gradient reaches any non-LoRA parameter or the text encoder; the teacher equals the original CLIP even after the adapter moved.
- loss scale and gradient norm are linear in lambda over 1e-3..1e1 for all three losses; raw loss = mean squared difference and is quadratic in the perturbation.
- reference-pool filters, pHash index (checked against brute force), pool hash, tamper detection, ZS-test probe lock.

## NOT tested (needs Colab, first run may need small fixes)
- `train.py main()` end-to-end with real CLIP, DTD/EuroSAT loaders, `probes.load_probe` on real Caltech101/Pets, `ReferencePool` with the real tokenizer.
- `build_reference_pool.py` against the real `pixparse/cc3m-wds` shards (naming `cc3m-train-XXXX.tar`, sample keys, `.txt/.json` fields), the `--source urls` path, and the torchvision eval-dataset hashing (`Food101` ~5 GB, exhaustive pHash takes a while).
- GPU determinism. The bit-for-bit result above is on **CPU**. On GPU we use eager attention + `use_deterministic_algorithms(warn_only=True)`; if a warning prints, bit-for-bit on GPU is not guaranteed and the claim must be restated as "same seed, same metrics within float tolerance".
- fp32 only (no AMP) so that the determinism claim is clean. A T4 timing pilot decides whether AMP is worth its cost.

## Findings that change the Phase 2 request (please read)
1. **ZSCL-CE and GNSP-KL are the same optimizer.** With a detached teacher, CE(teacher, student) = KL(teacher || student) + H(teacher), and H(teacher) is constant, so the gradients are identical (tested). The two "distributional baselines" therefore differ only in reported loss value, not in training behaviour, **if both use the same reference batches, same S, same direction and same weights**. Consequence: treat them as ONE baseline, "distributional distillation (softmax-normalized, row+col)". A genuinely different ZSCL variant needs ZSCL's other parts (its reference image/text sources, the weight-averaging in parameter space) and/or the literal `student_target` orientation (implemented as an option, flagged below). Do not present two near-identical curves as two baselines.
2. **CE is not zero at student == teacher.** It equals the teacher entropy. The requested test "0 for CE" is impossible by construction; the test asserts value == entropy and excess CE == 0.
3. **Notation conflict for tau.** The roadmap defines S = tau * cos with tau about 100 (CLIP's logit scale). Your request also writes softmax(S/tau); with S = tau * cos that equals softmax(cos), i.e. almost no temperature, which is not what CLIP/ZSCL/GNSP do. In GNSP, Z = S_cos / tau with tau = 0.01. I implemented all three losses on the same logit matrix `S = logit_scale * cos` (logit_scale = exp(CLIP logit_scale) = 100), softmax applied to S directly. This is equivalent to GNSP's cos/0.01. The only controlled difference is then how S is normalized: none (raw), softmax-CE, softmax-KL.
4. **transformers >= 5.20** returns a `BaseModelOutputWithPooling` from `get_image_features` / `get_text_features` (`pooler_output` is the projected embedding; verified numerically). The Phase 1 pin (5.18.0) returns a tensor. `lora.py` handles both. `frozen_eval.py` (Phase 1, frozen) does not: keep `transformers==5.18.0` pinned on Colab (`pip install transformers==5.18.0`) or the Phase 1 script will break.
5. The reference batch similarity matrix is N x N within a batch (image i with caption i on the diagonal; other captions act as the "M" texts). With `--ref-batch-size 64` that is 4096 entries. Reference batch size changes the loss; it is an experimental knob, fix it before comparing methods.

## Implementation choices to verify (marked [VERIFY] in code)
- ZSCL: orientation of CE and the text-direction term (the paper gives no explicit equation); ZSCL's own reference texts (the paper's Table 1 caption and Sec. 5 disagree). Check the official code.
- GNSP-CD: KL direction (teacher || student) is from your request; the fetched excerpt does not state it. Check Eq. 14-16.
- LoRA: A ~ kaiming-uniform(a = sqrt 5), B = 0, scaling alpha / r, default alpha = 2r, rank 4, q and v of all 12 vision blocks, no dropout. These are pilot defaults, not tuned (tune on DEV only). Trainable parameters at rank 4 = 12 layers x 2 modules x (4 x 768 + 768 x 4) = 147,456 (check in summary.json).
- Optimizer AdamW, constant lr 1e-4, batch 32, no schedule, no augmentation (protocol says none for baseline). All placeholders.

## Protocol interactions (changelog entry in docs/protocol_changelog.md)
- ZS-dev probes use a **fixed seeded subsample** (default 2000 images, hash logged). Caltech101 class names are cleaned torchvision names, not CLIP's exact list.
- `ZS-test` probes (cifar100, food101) are locked in code (`PermissionError`) and the trainer never evaluates the held-out TEST pools. The final-evaluation script (Phase 4) must unlock them explicitly and log the count.
- Reference pool: source is the `pixparse/cc3m-wds` mirror (images included), not the dead-prone URL list; caption filtering is **Porter-stem phrase matching** (deterministic, no downloads, conservative superset of lemma matching) with EuroSAT "a or b or c" class names split into separate phrases; probe class names are **flagged, not removed**, by default (`--probe-classname-mode remove` to enforce). Images stored as JPEG q95, shorter side <= 256.
- Near-duplicate check is pHash (Hamming <= 6) against DTD (all 3 splits), EuroSAT (all), Caltech101, Oxford Pets, CIFAR-100, Food-101 (all splits). pHash misses crops, heavy edits and re-photographs; state this limitation.
- NSFW: the mirror provides a flag only if present in the shard JSON; the manifest counts `nsfw_field_missing`. If it is always missing, the paper must say NSFW filtering was not performed.

## Pilot Run Findings (Phase 2B)
A 30-step pilot run on DTD (rank 4, batch 32, reference batch 64) revealed the following loss magnitudes:
*   `l_task` (cross-entropy) oscillates around **2.0**.
*   `raw_frobenius` (`lambda=1e-3`) grows to **~0.0008** by step 30.
*   `gnsp_cd` (`lambda=1.0`) grows to **~0.033** by step 30.

**Implication for Phase 3:** The initial lambda values were too small; the preservation loss was easily ignored by the optimizer. Phase 3 hyperparameter grids must be scaled up:
*   **Raw Frobenius grid:** `[0.0, 0.1, 1.0, 10.0]`
*   **GNSP/ZSCL grid:** `[0.0, 10.0, 100.0, 1000.0]`
*(Note: ZSCL and GNSP are mathematically equivalent optimizers; Phase 3 will compare `raw` against `gnsp` to save compute, as justified in finding 1 above).*