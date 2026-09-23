# Từ các thử nghiệm trước đến E45

Bảng này trả lời câu hỏi **mỗi giai đoạn giúp chốt điều gì**. Nó không phải bảng xếp hạng các phiên bản: các thử nghiệm E-series từng dùng mẫu, config và điều kiện chạy khác nhau. Chỉ so điểm khi đã đối chiếu cùng tập câu, scorer và artifact.

| Giai đoạn | Câu hỏi kỹ thuật | Quyết định đưa vào E45 | Code liên quan |
|---|---|---|---|
| E00/E02 | Có thể dựng lexical và dense index từ đúng corpus BTC không? | Giữ lineage của corpus, BM25 và dense index đã pin hash. | `e00_build.py`, `e02_dense.py`, config E45 |
| E03/E08A | Gộp hai nhánh retrieval và giữ thứ tự seed thế nào? | RRF trọng số bằng nhau, `k=60`; 40 candidate mỗi nhánh, 12 seed cho context. | `e03_rrf_grid.py`, config E45 |
| E21 | Mở rộng chunk về Điều luật mà vẫn truy được span nguồn ra sao? | Dùng parent context với fallback có giới hạn và span khớp văn bản gốc. | `e21_parent_context.py` |
| E38 | Baseline LoRA nào sẽ được giữ nguyên để so với thử nghiệm tiếp theo? | Đóng băng E38 làm control; không đổi adapter và acceptance gate khi thử E45. | `e38_viqwen_metadata_lora.py`, mục `control` trong config |
| E43/E44 và P00/P01 | Chốt prompt, cleanup và giới hạn sinh nào? | Prompt parent context; greedy 1.024 token; sinh lại đến 1.536 token nếu dừng vì độ dài; chỉ cleanup phần đuôi. | `final_public_e43.py`, `final_public_e44.py`, `final_private_p00.py`, `final_private_p01.py` |
| E45 | Fine-tune trên cùng kiểu evidence và prompt dùng lúc inference có hợp lý hơn không? | Train một LoRA rank 8 mới trên 5.636 record chính thức, `assistant-only loss`, 705 optimizer step. | `e45_parent_training.py`, script train, config |
| Bản nộp private | Khi bốn shard và thời hạn thực tế gặp nhau, output nào đã được nộp? | Ghép checkpoint đã lưu; 143 lượt sinh lại chưa xong dùng output lượt đầu. | `e45_private_*.py`, script merge khẩn cấp, `submitted-run.md` |

Giả thuyết của E45 là **đồng bộ evidence và prompt giữa train với inference**. Điểm private của bản nộp không tự chứng minh E45 tốt hơn E38 trên một paired test có kiểm soát. Repo giữ source của các thí nghiệm cũ để người đọc lần theo quyết định; record đánh giá và đáp án tham chiếu không được public.

Ba con số dễ nhầm: **200 câu** là holdout để thiết kế phép kiểm tra local, **5.636 câu** dùng để train E45, còn **1.918 câu** thuộc bản nộp private. Chúng nằm ở ba giai đoạn khác nhau.
