# Các quyết định kỹ thuật dẫn đến bản nộp

Đây là lịch sử **chọn kỹ thuật**, không phải bảng xếp hạng các phiên bản. Các thử nghiệm dùng mẫu, config và điều kiện chạy khác nhau; chỉ so điểm khi đã đối chiếu cùng tập câu, scorer và artifact.

| Thành phần | Câu hỏi đã kiểm tra | Quyết định của bản nộp |
|---|---|---|
| Corpus và index | Có thể xây lexical và dense retrieval từ đúng corpus chính thức không? | Chuẩn hóa document/chunk, lưu BM25 trong SQLite, tạo dense vector index bằng `AITeamVN/Vietnamese_Embedding` và pin hash của từng artifact. |
| Hybrid retrieval | Gộp hai danh sách kết quả thế nào để không cộng trực tiếp BM25 score với vector similarity? | Dùng RRF với trọng số ngang nhau và `k=60`; lấy 40 candidate mỗi nhánh, 20 kết quả sau fusion, 12 seed cho context. |
| Context theo Điều luật | Làm sao đưa đủ ngữ cảnh pháp lý mà vẫn truy ngược được về văn bản gốc? | Mở rộng seed về Điều luật hoặc đoạn lân cận theo giới hạn; giữ thứ tự seed, span nguồn và fallback. |
| Baseline fine-tune | Cần mốc đối chiếu nào trước khi đổi cách chuẩn bị dữ liệu train? | Giữ một LoRA rank 8 đã train trước đó làm control; adapter và acceptance gate được đóng băng trong kế hoạch thí nghiệm. |
| Prompt và generation | Prompt, decoding và cleanup nào được giữ cố định? | Prompt gồm câu hỏi và parent context; greedy tối đa 1.024 token, sinh lại đến 1.536 token chỉ khi dừng vì độ dài; cleanup phần đuôi. |
| Fine-tune đồng bộ với inference | Model có nên học trên đúng kiểu evidence và prompt dùng lúc trả lời? | Train một LoRA rank 8 mới trên 5.636 record chính thức, `assistant-only loss`, 705 optimizer step, cùng renderer context/prompt khi inference. |
| Bản nộp sát deadline | Nếu lượt sinh lại chưa xong, output nào thực sự được nộp? | Giữ lượt sinh lại đã hoàn tất; dùng output lượt đầu đã checkpoint cho 143 trường hợp còn lại. |

**Giả thuyết chính** là đồng bộ evidence và prompt giữa train với inference sẽ giảm độ lệch đầu vào của generator. Điểm private của bản nộp **không tự chứng minh** LoRA mới tốt hơn baseline trên paired test có kiểm soát. Repo giữ source của các thử nghiệm để truy lại quyết định; record đánh giá và đáp án tham chiếu không được public.

Ba con số dễ nhầm: **200 câu** là holdout dùng trong kế hoạch kiểm tra local, **5.636 câu** dùng để train, còn **1.918 câu** thuộc bản nộp private. Chúng nằm ở ba giai đoạn khác nhau.

## Mã thí nghiệm còn trong source

Tên file và trường trong config vẫn giữ mã của lần chạy đã nộp để kiểm tra hash/checkpoint. Mã này chỉ phục vụ truy nguồn, không phải tên công nghệ:

| Vai trò kỹ thuật | Mã trong source | File tiêu biểu |
|---|---|---|
| Chuẩn hóa corpus, SQLite BM25 index, dense vector index | E00, E02 | `e00_build.py`, `e02_dense.py` |
| RRF và thứ tự seed retrieval | E03, E08A | `e03_rrf_grid.py`, config retrieval |
| Mở rộng chunk thành context theo Điều luật | E21 | `e21_parent_context.py` |
| LoRA control trước lần train cuối | E38 | `e38_viqwen_metadata_lora.py` |
| Prompt, greedy decoding và cleanup phần đuôi | P00/P01, E43/E44 | `final_private_p00.py`, `final_private_p01.py`, `final_public_e43.py`, `final_public_e44.py` |
| LoRA train trên context/prompt đồng bộ với inference | E45 | `e45_parent_training.py`, config và script train |

Không đổi những tên này trong code chỉ để đẹp README: chúng nằm trong identity và hash của artifact đã dùng khi thi.
