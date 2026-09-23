# UIT DSC 2026 Task 2 — recorded rules and engineering implications

This is the team's recorded technical interpretation of the organizer's Task 2 materials and scoring program, not a substitute for the organizer's current rules. The original source documents are not redistributed here.

| Rule or contract recorded during the competition | E45 implication |
|---|---|
| Use organizer-provided training data and official legal corpus; do not add external corpus or synthetic QA, answers, evidence or negatives | E45 training uses 5,636 selected official records and official corpus-derived contexts only. No external legal API or generated supervision is in the pipeline. |
| Total parameters across generator, embedding and any other model in one Task 2 system must be **strictly below 4 billion**; quantization and LoRA do not reduce the counted base-model parameters | Frozen inventory reports 3,668,660,224 parameters. The repo does not claim NF4 or LoRA reduces the official count. |
| Models must be downloadable and run under team control; model APIs and intermediary AI products are disallowed | Generator and dense encoder are pinned to immutable Hugging Face revisions and run locally on Kaggle GPUs. |
| Model name, URL, parameter inventory, license and registration status must be recorded for the organizer | The config records identities/counts. Registration status is not evidenced in this public snapshot and must not be inferred from the score. |
| Primary score is METEOR; secondary score is ROUGE-L | The recorded scoring-program checksum is `4fac914203d325445a666c0c566530c962ba95b843e1988e4f37057c47447891`. Its METEOR tokenization is whitespace based; vendored ROUGE-L uses ASCII-normalized tokens. |
| Codabench submission contains only `submission.json` in a ZIP | The scorer actually requires a JSON object mapping ID to exactly `{"answer": string}`. The submitted ZIP passed the local format validator. |
| Official data and private references require isolation | This public repository excludes full dataset, private questions/answers, generated answer payloads, indexes and checkpoints. |

Raw corpus context fields were `id`, `link`, `passage`, with optional `name`. The audited canonical corpus revision was SHA-256 `9a4441b4537ceb646b15359f470a1da0904e6c92a61e8c4c376c19e17dec395e`. The actual source scorer may have changed during the competition; these notes describe the checked artifact only. No private scoring references are included.
