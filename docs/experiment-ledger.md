# Experiment and decision ledger

This is a map of the frozen source lineage, not a table of comparable leaderboard scores. Earlier E-series experiments used different conditions; their outcomes should not be compared numerically without the same sample, scorer and artifacts.

| Stage | Question answered | Decision carried into E45 | Evidence location in this repo |
|---|---|---|---|
| E00/E02 | Can official passages be indexed for lexical and semantic retrieval? | Retain exact official corpus lineage, BM25 and the pinned dense index. | `src/uit_dsc_fixed_rag/e00_build.py`, `e02_dense.py`, frozen config hashes |
| E03/E08A | How should the two retrieval branches and ordered seeds be combined? | Equal-weight RRF (`k=60`), 40 candidates per branch, 12 context seeds. | `e03_rrf_grid.py`, E45 config |
| E21 | Can a seed expand to its legal parent without losing source spans? | Parent-context expansion with bounded fallbacks and exact source spans. | `e21_parent_context.py` |
| E38 | What frozen LoRA control should the next experiment compare against? | Keep E38 as control; do not change its adapter or the local paired acceptance gate. | `e38_viqwen_metadata_lora.py`, `control` section of config |
| E43/E44 + P00/P01 | What inference prompt, cleanup and length policy will be frozen? | Parent-context prompt, greedy 1,024-token pass, 1,536-token length-only restart and suffix cleanup. | `final_public_e43.py`, `final_public_e44.py`, `final_private_p00.py`, `final_private_p01.py` |
| E45 | Can training use the same expanded evidence and prompt format as inference? | One fresh rank-8 LoRA on 5,636 official records, assistant-only loss, 705 optimizer steps. | `e45_parent_training.py`, training script and frozen config |
| Private submission | How was the trained candidate actually submitted? | Four shards, saved per-row checkpoints and a 143-row deadline fallback. | `e45_private_*.py`, emergency merger and `submitted-run.md` |

The E45 hypothesis is about **training/inference evidence alignment**. It does not establish that every earlier experiment was inferior. The private METEOR/ROUGE-L pair in README is a final-submission result, not a controlled E38-vs-E45 paired delta. Source code for intermediate studies is retained to make the decision chain inspectable; raw evaluation records and reference answers are withheld.

The 200-question holdout was used for local evaluation design. The 5,636 rows refer to training; the 1,918 questions refer to the private submission. These are different datasets and phases.
