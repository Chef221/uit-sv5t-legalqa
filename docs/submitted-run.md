# Bản private đã nộp

File ZIP cuối có SHA-256 `d697c48e4ce48410ad4695291c98891f9f872a88cf0f6d1dfc576d8dbf86496b`. `submission.json` bên trong có SHA-256 `5232019db849feac81182aa7f870941886a9812b689aa8015dc56c3461d6bbdf`. Bản nộp có đúng 1.918 ID, mỗi ID ứng với một answer string; bước kiểm tra local đã xác nhận format và coverage. Payload không được đưa vào repo.

Archive của LoRA candidate cuối có SHA-256 `f769add107c01347923a697237266c10e32a10dee3b0d57823f5708a26157b67`. Bốn checkpoint được dùng để ghép bản nộp:

| Shard | SHA-256 checkpoint | Lượt sinh lại hoàn tất | Câu dùng output lượt đầu |
|---:|---|---:|---:|
| 0 | `52a2d19f2be7fc9a25e06c497fcf307b36f5da7b1eac1ee4a883868b3e0d2764` | 52 | 33 |
| 1 | `481c9924f3b14fecb0de5da541b60f7cb8f044c0ef10d0ead228bb3ab0418485` | 45 | 28 |
| 2 | `5296da2909708fbbd0ff5313acf0458c5de3e8c3e9a0ba926131568ad8ca9815` | 45 | 36 |
| 3 | `d1edbee1f9f1cd19c76dc93b49ff49227101e1d3c85ef336e4960d2ee8064de4` | 41 | 46 |

Tổng cộng, merger giữ **183** lượt sinh lại đã xong. **143** trường hợp còn dang dở dùng answer 1.024 token ở lượt đầu đã lưu trong checkpoint. Đây là cách ghép **riêng của lần nộp sát deadline**, không phải lời khẳng định rằng mọi lượt sinh lại đã chạy xong.

Điểm đội nhận cho bản private: METEOR `0.597327402`, ROUGE-L `0.586192885`. Repo chưa có bản export kết quả Codabench lưu lâu dài để đối chiếu độc lập với đúng SHA-256 của ZIP. Báo cáo merge đầy đủ có ID câu hỏi private nên không được public; trang này chỉ giữ số liệu tổng và hash. Merger không đọc đáp án tham chiếu private.
