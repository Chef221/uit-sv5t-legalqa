# Reproduction guide

This is a **source-only** release. Authorized users must obtain the organizer's task data and scoring program through official channels. Download the pinned base and embedding models at the immutable revisions in `configs/e45-inference-aligned-parent-lora-v1.json`; obtain or rebuild E00/E02 artifacts under the hashes in `artifact-hashes.md`. Never place inputs under Git tracking.

The submitted path used:

1. `scripts/run_e45_static_prepare.py` to check the frozen config and materialize the 5,636 answer-supervised training rows.
2. `scripts/run_e45_train_kaggle.py` through `notebooks/E45-ACCOUNT-A-RESUME-AND-DIRECT-PRIVATE-PACKAGE-T4X2.ipynb`, with two T4 GPUs and DDP world size two, to finish the 705-step candidate archive.
3. Four `notebooks/E45-PRIVATE-RESUMABLE-SHARD-*-OF-4-T4X2.ipynb` instances to prepare private contexts and generate per-shard checkpoints. Their archive and admission hashes must match before model loading.
4. `scripts/merge_e45_emergency_checkpoints.py` to assemble the **as-submitted deadline fallback**. The standard all-restarts-complete merger is a different policy and cannot be used to claim the published private score.

The exact Kaggle Python package versions are recorded in the config's `runtime` section. GPU/kernel driver behavior, downloaded model snapshots and external artifacts still have to be verified on the target runtime. Do not interpret passing CPU tests as proof of a complete GPU rerun.

For static checks, create a Python environment with `pytest` installed and run:

```bash
python -m compileall -q src scripts
python -m pytest -q tests/test_e45_static_prepare_jsonl.py tests/test_private_checkpoint_resume.py
```

Test modules that need training frameworks, organizer data or CUDA are intentionally outside that small default command. The data boundary is strict: input resolution is by immutable SHA-256, duplicate matches are rejected, and any private reference field must fail before generation or finalization.
