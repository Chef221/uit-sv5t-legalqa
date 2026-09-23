# Kiến trúc hệ thống Hybrid RAG

[Sơ đồ tổng quan](architecture-overview.svg) cho thấy luồng từ câu hỏi đến `submission.json`. Hệ thống tách ba phần: chuẩn bị corpus và dữ liệu train ở offline; retrieval và dựng context khi inference; generation và đóng gói bản nộp.

## Chuẩn bị offline

Corpus chính thức được chuẩn hóa thành document và chunk, có metadata để truy ngược về văn bản nguồn. **SQLite BM25 index** hỗ trợ tìm kiếm theo từ khóa. **Dense vector index** được tạo bằng `AITeamVN/Vietnamese_Embedding` (vector 1.024 chiều, 457.340 record); artifact lưu dạng FAISS. Ở lần chạy đã nộp, truy vấn dense dùng exact inner-product search trên GPU với vector chuẩn hóa. Mỗi artifact được kiểm tra SHA-256 trước khi dùng lại.

Để fine-tune, hệ thống lấy thứ tự seed retrieval đã chốt cho **5.636 câu train**. Từng seed được mở rộng về Điều luật hoặc đoạn lân cận theo chính sách có giới hạn: tối đa 1.200 token cho parent, giữ span chính xác trong văn bản nguồn và không tự loại seed để nhường chỗ cho expansion. Cùng renderer tạo prompt cho train và inference. **Đáp án chính thức không tham gia chọn evidence**; nó chỉ đi vào target của LoRA, với `assistant-only loss` và một EOS sau câu trả lời. Tổng sequence không vượt 8.192 token.

Repo chỉ chứa code và config. Index, record train đã materialize và LoRA weights nằm ngoài Git.

## Retrieval và dựng prompt khi inference

Với mỗi câu hỏi, BM25 và dense retrieval lấy **40 candidate mỗi nhánh**. **Reciprocal Rank Fusion (RRF)** trộn hai danh sách với trọng số 0,5 / 0,5 và hằng số `k=60`. Hệ thống giữ top 20 sau fusion rồi chọn 12 seed context. Không có cross-encoder reranker trong cấu hình này.

Mỗi seed được mở rộng sang ngữ cảnh Điều luật, có fallback theo giới hạn token và giữ lại vị trí nguồn. Prompt ghép câu hỏi với evidence đã chọn; cùng một kiểu prompt được dùng khi materialize dữ liệu fine-tune. Đây là quyết định chính của bản chạy: giảm độ lệch giữa context model thấy lúc học và context nó nhận khi trả lời.

## Model và generation

Generator là `AITeamVN/Vi-Qwen2-3B-RAG` cùng **LoRA rank 8** đã train trên dữ liệu chính thức. Generation dùng greedy decoding: lượt đầu tối đa **1.024 token mới**. Chỉ câu dừng vì chạm trần độ dài mới được sinh lại từ **input IDs gốc**, với giới hạn **1.536 token**. Cleanup xử lý các đoạn lặp ở đuôi câu trả lời, không cắt nội dung giữa câu.

Prompt ngắn được chia cho **hai worker độc lập**, mỗi worker dùng một T4. Prompt dài chạy trên worker chia model qua hai T4. Bốn shard private có thể chạy ở bốn tài khoản Kaggle khác nhau. Sau mỗi câu hoàn thành, checkpoint lưu output cùng hash của input, model, adapter và runtime.

## Ranh giới tin cậy

Artifact được tìm theo SHA-256, không chọn theo tên file hoặc đường dẫn upload. Thiếu file hoặc có nhiều bản cùng hash đều bị từ chối. Worker state và record đầu ra có identity để phát hiện trộn shard hoặc adapter. Private inference không nhận đáp án tham chiếu làm input; chấm điểm local là bước riêng.

[Báo cáo bản nộp](submitted-run.md) ghi rõ 143 câu dùng output lượt đầu vì các lượt sinh lại chưa xong trước deadline. Đó là thuộc tính của **lần nộp thực tế**, không phải chính sách inference đầy đủ đã định.
