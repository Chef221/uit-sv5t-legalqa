# Submitted system architecture

## Offline boundary

Official BTC passages are normalized and chunked into E00 documents/chunks and a SQLite BM25 index. A pinned `AITeamVN/Vietnamese_Embedding` encoder produces the E02 dense index. E08A seed retrieval, E21 parent expansion and E44 prompt construction materialize answer-independent evidence for 5,636 official training examples. The E45 rank-8 LoRA trains only on the official answer target, with assistant-only loss and an 8,192-token sequence envelope. Indexes, training rows and adapter weights are external artifacts and are not part of this repository.

## Inference boundary

For each question the fixed retriever takes 40 BM25 and 40 dense candidates, combines ranks using equal-weight reciprocal rank fusion (`k=60`), retains 20 fused candidates and passes the first 12 seeds to parent-context expansion. The selected evidence is rendered using the frozen P00/P01 prompt. The generator is the pinned Vi-Qwen2-3B-RAG base plus the E45 adapter, using greedy decoding. The normal first pass allows 1,024 new tokens; length-only endings qualify for a fresh generation from the original input IDs with a 1,536-token cap. E43/E44 cleanup produces the answer string.

Short prompts are split between two isolated T4 workers; long prompts use a two-T4 sharded worker. The four private question shards can run on separate Kaggle accounts. Per-row checkpoints and SHA-256 bindings protect the output against a truncated session or accidental arm mix-up. The actual deadline assembly is documented in [submitted-run.md](submitted-run.md).

## Trust and provenance boundaries

The code resolves artifacts by immutable hash and rejects missing or multiple matches. The raw/clean record and worker states bind generation to prompt, context, model, adapter, runtime and decoding identities. Reference answers are not inputs to private inference. Local development evaluation was kept separate from private generation. The final emergency merger consumed saved checkpoint outputs and did not read private reference answers.
