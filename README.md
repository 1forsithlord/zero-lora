# Zero-LoRA

Similarity-drift-regularized LoRA adaptation of CLIP ViT-B/16 for zero-shot retention.
Team: Yash Pratap Singh, Shreyash Joshi, Yash Raj. DY Patil University, Ambi, Pune.

**Primary question (RQ1):** does raw image-text similarity-geometry regularization improve the trade-off between downstream task accuracy and zero-shot retention versus plain LoRA and distributional feature distillation? Effectiveness and novelty are hypotheses; nothing in this repo is a result until it is produced by a logged run.

## Status: Phase 1 (Weeks 1-2)
| Deliverable | File | State |
|---|---|---|
| Literature notes | `docs/literature_notes.md` | drafted; MoE-Adapters and LoRA-merging entries need full reads |
| Novelty matrix, claims ledger, gate | `docs/novelty_matrix.md` | drafted; team to confirm verdict |
| Data protocol v1 | `configs/protocol_v1.yaml` | DRAFT; freeze on Day 4 after team sign-off |
| Leakage checklist | `docs/leakage_audit_checklist.md` | ready |
| Frozen-CLIP eval | `src/zerolora/frozen_eval.py` | written, **not yet run** (needs GPU + internet) |
| Baseline CSV | `results/baseline_frozen_clip.csv` | **does not exist yet**; produced by the command below |

## Layout
```
configs/    protocol_v1.yaml (pools, prompts, preprocessing, seeds)
src/zerolora/   prompts.py, splits.py, frozen_eval.py   (later: lora.py, losses/, drift.py)
scripts/    run_frozen_eval.py
tests/      test_splits.py
docs/       literature_notes.md, novelty_matrix.md, leakage_audit_checklist.md
results/    baseline_frozen_clip.csv, run_manifest.json, leakage_audit_log.csv
notebooks/  phase1_colab.ipynb, phase1_kaggle.ipynb
```

## Reproduce the Phase 1 baseline (Kaggle, GPU T4, Internet ON)
```bash
pip install -q transformers pyyaml
python tests/test_splits.py                      # split logic, no GPU needed
python scripts/run_frozen_eval.py --config configs/protocol_v1.yaml \
       --root /kaggle/working/data --out results --check-rerun
```
Outputs `results/baseline_frozen_clip.csv` (dataset, pool, prompt mode, n, top-1, mean per-class accuracy, split hash, model id/commit) and `results/run_manifest.json` (python/torch/torchvision/transformers versions, CUDA, GPU, HF model commit, preprocessing, seeds, split-index hashes, wall-clock).

**Gate 1 check:** `--check-rerun` runs the evaluation twice and exits non-zero if any prediction differs. After the first successful run, copy the model commit into `protocol_v1.yaml: base_model.revision` and the library versions into `requirements.txt`, then commit and tag `protocol-v1`.

If dataset downloads fail on Kaggle (torchvision hosts), attach DTD and EuroSAT as Kaggle datasets, point `--root` at them, and record the dataset source/version in the manifest by hand.

## Known open items before the freeze
1. ~~Verify prompt strings~~ done 2026-10-16 (matches openai/CLIP prompts.md).
2. Confirm ZS-dev / ZS-test probe datasets (YAML proposals only).
3. Build the CC3M reference pool (3-day time box); not started.
4. Confirm the novelty-gate verdict in `docs/novelty_matrix.md`.
5. The code has been syntax-checked and the split logic unit-tested in the authoring sandbox only; the CLIP pipeline itself has not been executed (no torch/GPU/dataset access there). Expect small fixes on first Kaggle run.
