# Nội quy Task 2 và cách hệ thống áp dụng

Trang này tóm tắt những điều đội đã kiểm tra trong tài liệu và scoring program của UIT DSC 2026 Task 2. Đây là ghi chép kỹ thuật của đội, không thay thế thông báo chính thức của Ban tổ chức. Source tài liệu/scorer của BTC không được đưa vào repo.

| Quy định hoặc format đã kiểm tra | Cách hệ thống xử lý |
|---|---|
| Chỉ dùng dữ liệu train và corpus chính thức; không thêm corpus ngoài, synthetic QA, answer, evidence hay hard negative | LoRA train trên 5.636 record chính thức; context lấy từ corpus BTC. Pipeline không gọi legal API ngoài. |
| Tổng số tham số của mọi model trong hệ thống Task 2 phải **nhỏ hơn 4 tỷ** | Bản inventory đã chốt ghi 3.668.660.224 tham số. Quantization và LoRA không được dùng để trừ số tham số của base model khi tính ngưỡng. |
| Model phải tải và chạy dưới quyền kiểm soát của đội; không dùng model API hay sản phẩm AI trung gian | Generator và embedding model được pin theo revision, chạy trực tiếp trong môi trường GPU của đội. |
| Phải ghi tên, URL, số tham số, license và trạng thái đăng ký model | Config có identity và số tham số. Repo này chưa có bằng chứng độc lập về việc đăng ký model, nên không suy ra trạng thái đó từ điểm thi. |
| METEOR là metric chính; ROUGE-L là metric phụ | Scoring program đã kiểm tra có SHA-256 `4fac914203d325445a666c0c566530c962ba95b843e1988e4f37057c47447891`. METEOR dùng token tách theo khoảng trắng; bản ROUGE-L vendored chuẩn hóa token về ASCII. |
| Codabench nhận ZIP chứa `submission.json` | Scorer thực tế yêu cầu JSON object dạng `question_id -> {"answer": string}`. Bản nộp đã qua kiểm tra format và coverage ở local. |
| Không để đáp án tham chiếu đi vào private inference | Code tách reference khỏi đầu vào sinh; repo public cũng không chứa câu hỏi/đáp án private. |

Corpus gốc dùng `id`, `link`, `passage`; `name` có thể không có. Revision canonical đã audit có SHA-256 `9a4441b4537ceb646b15359f470a1da0904e6c92a61e8c4c376c19e17dec395e`. Nếu BTC phát hành data hoặc scorer khác, cần kiểm tra lại checksum và contract trước khi áp dụng các kết luận ở đây.
