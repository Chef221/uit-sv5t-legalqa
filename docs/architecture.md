# Kiến trúc của bản nộp E45

[Sơ đồ tổng quan](architecture-overview.svg) cho thấy luồng từ câu hỏi đến `submission.json`. Trang này đi sâu vào ranh giới giữa chuẩn bị dữ liệu, train và inference.

## Chuẩn bị offline

Corpus do BTC cung cấp được chuẩn hóa và chia thành document/chunk E00. Từ đó hệ thống tạo SQLite BM25 index. Model `AITeamVN/Vietnamese_Embedding` ở revision đã chốt tạo dense index E02. Cả hai index đều được kiểm tra bằng SHA-256 khi dùng lại.

E08A lưu thứ tự seed retrieval cho 5.636 câu train. E21 mở rộng mỗi seed về `parent context`, có giới hạn độ dài và giữ span trỏ ngược về văn bản nguồn. Prompt dùng cách render đã chốt ở E44. Bước này không đọc câu trả lời để quyết định evidence. Câu trả lời chính thức chỉ đi vào target của E45 LoRA: `assistant-only loss`, một EOS sau answer và giới hạn 8.192 token cho toàn bộ sequence.

Repo chỉ chứa code và config. Index, record train đã materialize và adapter weights nằm ngoài Git.

## Inference

Với mỗi câu hỏi, BM25 và dense retrieval lấy 40 candidate mỗi nhánh. RRF trộn hai danh sách với trọng số 0,5 / 0,5 và `k=60`; hệ thống giữ top 20 sau fusion và đưa 12 seed đầu vào E21. Context sau mở rộng được render thành prompt P00/P01 cho Vi-Qwen2-3B-RAG cùng adapter E45.

Generation chạy greedy. Lượt đầu tối đa 1.024 token mới. Chỉ trường hợp dừng vì chạm giới hạn độ dài mới được sinh lại, bắt đầu từ **input IDs gốc**, với giới hạn 1.536 token. E43/E44 xử lý lặp ở phần đuôi sau khi model sinh xong.

Prompt ngắn được chia cho hai worker độc lập, mỗi worker dùng một T4. Prompt dài dùng worker chia model trên hai T4. Bốn shard private có thể chạy trên các tài khoản Kaggle khác nhau. Mỗi câu hoàn thành được ghi vào checkpoint, kèm các hash ràng buộc input, model, adapter, runtime và output.

## Ranh giới tin cậy

Artifact được tìm theo SHA-256, không chọn theo tên file hay đường dẫn upload. Thiếu file hoặc có nhiều bản cùng hash đều bị từ chối. Worker state và record đầu ra ghi đủ identity để phát hiện trộn shard hoặc trộn adapter. Private inference không nhận đáp án tham chiếu làm input. Việc chấm local được tách khỏi luồng sinh private.

[Báo cáo bản nộp](submitted-run.md) ghi rõ ngoại lệ 143 câu ở deadline. Ngoại lệ này là thuộc tính của **lần nộp cuối**, không phải chính sách inference lý tưởng đã định trước.
