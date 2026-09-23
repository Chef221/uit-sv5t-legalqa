# Hash của các artifact nằm ngoài Git

Các giá trị dưới đây chỉ để nhận diện đúng phiên bản. Repo **không chứa** những file này.

| Artifact | SHA-256 |
|---|---|
| Corpus chính thức, revision canonical | `9a4441b4537ceb646b15359f470a1da0904e6c92a61e8c4c376c19e17dec395e` |
| Manifest của corpus đã chuẩn hóa | `04efd3905ad6d2758461587ca68d7d70fa2c568855b78f0c762c7a37ba547b2e` |
| SQLite BM25 index | `f874a9528433f0db64efe7b3e951028d89433cbfb4c6e951a182b531acf1e0f0` |
| Chunk metadata | `1e48c7762765ac2dd169045e9f5327c5311db3f1da8a6007ef70fff58718e367` |
| Document metadata | `f2968724e8a25124034b9ff2144427f8853ed37359443bd149ec3832f4c1fed7` |
| FAISS dense vector index | `96ab6b8afcb376e327116642e0ffc378d633d9f712b698afd0e43dcee654a633` |
| Ánh xạ ID của dense vector index | `6e26962a0963f50460ada74707db31e604dfe4991e7d7150afeec89aa363fb99` |
| Record train đã materialize | `84fb204f6268ebd4b0a56f1e4c5048c31a4e91778b9ac3c261d9de509a5ca17f` |
| Archive LoRA candidate cuối | `f769add107c01347923a697237266c10e32a10dee3b0d57823f5708a26157b67` |
| ZIP đã nộp | `d697c48e4ce48410ad4695291c98891f9f872a88cf0f6d1dfc576d8dbf86496b` |

Hash của bốn checkpoint private có trong [báo cáo bản nộp](submitted-run.md). Scoring program của BTC mà đội đã audit có SHA-256 `4fac914203d325445a666c0c566530c962ba95b843e1988e4f37057c47447891`.
