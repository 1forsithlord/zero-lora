# Leakage-audit checklist (protocol v1)

Run before every Gate and before any number is reported. Record the outcome (pass / fail / not-checked + note) in `results/leakage_audit_log.csv`. A failed or unchecked item is stated as a limitation in the paper.

## A. Pool disjointness
- [ ] Adaptation, dev and test pools of each task are disjoint by **index** (assert in code; `tests/test_splits.py`).
- [ ] Split indices are generated from `global_seed` and their hash is logged; a rerun reproduces the same hash.
- [ ] No test sample influences any choice: lambda, learning rate, rank, epochs, early stopping, prompts, class-name wording, checkpoint selection. (Frozen-CLIP numbers on test are allowed because nothing is tuned on them; still, list them as "observed".)
- [ ] Test is evaluated once per final configuration; the evaluation count is logged.

## B. Cross-dataset overlap
- [ ] Near-duplicate check (perceptual hash, e.g. pHash with Hamming threshold, plus exact hash) between: adaptation vs dev/test of every probe dataset; reference pool vs all adaptation/dev/test/probe images.
- [ ] Known dataset overlaps noted (e.g. same source images across datasets). State what was and was not checked.
- [ ] Zero-shot probe datasets used for ZS-dev are never used for ZS-test.

## C. Reference pool (CC3M subset)
- [ ] Captions containing any class name of any task or probe dataset removed or flagged; list the matching rule (exact, lemma, substring) and the count removed.
- [ ] Exact + near duplicate images removed; corrupt/unavailable URLs dropped; counts logged.
- [ ] Pool built once, hash logged, then frozen. Time box: 3 days; if exceeded, freeze the clean subset and document limits.
- [ ] Reference images never appear in any evaluation pool.
- [ ] Caveat recorded: CLIP's own pretraining data may overlap public web sets; absolute leakage from pretraining cannot be ruled out.

## D. Prompts and class names
- [ ] Templates and class-name strings frozen in `src/zerolora/prompts.py` before any dev evaluation; changes require a protocol changelog entry.
- [ ] Class names were not edited after looking at test errors.

## E. Teacher / target handling
- [ ] Teacher (frozen CLIP) and reference similarity targets are detached; no gradient to the teacher (unit test).
- [ ] Text encoder frozen in the primary experiment (assert requires_grad False).
- [ ] lambda = 0 reproduces plain LoRA bit-for-bit under a fixed seed (Phase 3 sanity check).

## F. Reporting
- [ ] Absolute task accuracy and absolute zero-shot accuracy always shown; retention ratio never alone.
- [ ] All runs kept, including failures and every seed.
- [ ] Limitations of this audit stated (what could not be checked).
