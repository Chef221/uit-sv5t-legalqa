# As-submitted private run

The actual submitted ZIP is identified by SHA-256 `d697c48e4ce48410ad4695291c98891f9f872a88cf0f6d1dfc576d8dbf86496b`; its internal `submission.json` SHA-256 is `5232019db849feac81182aa7f870941886a9812b689aa8015dc56c3461d6bbdf`. It contains exactly 1,918 ID-to-answer mappings and passed local format/coverage validation. Neither payload is distributed here.

The direct E45 candidate archive SHA-256 is `f769add107c01347923a697237266c10e32a10dee3b0d57823f5708a26157b67`. The four checkpoint archive hashes, in shard order, are:

1. `52a2d19f2be7fc9a25e06c497fcf307b36f5da7b1eac1ee4a883868b3e0d2764`
2. `481c9924f3b14fecb0de5da541b60f7cb8f044c0ef10d0ead228bb3ab0418485`
3. `5296da2909708fbbd0ff5313acf0458c5de3e8c3e9a0ba926131568ad8ca9815`
4. `d1edbee1f9f1cd19c76dc93b49ff49227101e1d3c85ef336e4960d2ee8064de4`

The deadline merger used all available completed second passes (52, 45, 45, 41 by shard). For 143 unfinished length restarts (33, 28, 36, 46 by shard), it selected each saved 1,024-token first-pass answer. This is a **run-specific fallback**, not the fully completed P01 policy. The private score reported by the team for this submission is METEOR `0.597327402`, ROUGE-L `0.586192885`. A durable Codabench result export and submission linkage are still required to independently substantiate that pairing.

Private question IDs and answer text are intentionally omitted. The merger's full report contains question IDs and stays outside Git; this document preserves only aggregate counts and hashes.
