# Protocol changelog (append-only)

## v1.2-proposed (Phase 2) -- needs team sign-off before the first dev-tuned run
`configs/protocol_v1.yaml` is NOT edited. The following are additions/clarifications that live in code and are recorded here (leakage checklist D):
1. **ZS-dev probe wording.** Caltech101: CLIP's 34 templates (verified against openai/CLIP prompts.md 2026-10-16); class names = torchvision categories, lower-cased, `_` -> space, with CLIP's renamings for Faces, Faces_easy, Leopards, Motorbikes. Oxford Pets: template `a photo of a {}, a type of pet.`, torchvision class names. Probes evaluated on a fixed seeded subsample (default 2000, seed = global_seed, hash logged in summary.json).
2. **ZS-test lock.** `cifar100`, `food101` cannot be loaded without `unlock_test=True`; trainer never unlocks.
3. **Reference pool details.** Source `pixparse/cc3m-wds` shards (ids recorded in manifest); Porter-stem class-name filter; probe class names flagged not removed (default); JPEG q95 <= 256 px; pHash threshold 6 against all eval and probe images; pool hash = sha256 of sorted (path, caption); image-content hash added.
4. **Similarity scale.** All preservation losses operate on S = logit_scale * cos with logit_scale = exp(CLIP logit_scale) = 100.
5. **Reference batch.** N x N within-batch similarity matrix; `ref_batch_size` default 64; reference loader has its own RNG stream.
6. **Baseline definition.** ZSCL-CE and GNSP-KL are optimization-equivalent (see phase2_notes.md, finding 1); the head-to-head is raw vs distributional (softmax-normalized) matching.

Sign-off: ____ Yash Pratap Singh   ____ Shreyash Joshi   ____ Yash Raj     Date: ________