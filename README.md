# E45 — Hệ thống hỏi đáp pháp luật tiếng Việt tại UIT DSC 2026

Đây là source của hệ thống đội đã nộp cho Task 2, UIT Data Science Challenge 2026. E45 tìm căn cứ trong kho văn bản do Ban tổ chức cung cấp, mở rộng các đoạn tìm được về ngữ cảnh của Điều luật, rồi dùng Vi-Qwen2-3B-RAG với LoRA rank 8 để viết câu trả lời.

<p align="center">
  <img src="docs/architecture-overview.svg" alt="Sơ đồ dọc của hệ thống E45, từ dữ liệu chính thức và câu hỏi đến retrieval, parent context, Vi-Qwen LoRA và submission" width="100%">
</p>

## Kết quả private test

| Metric | Điểm |
|---|---:|
| **METEOR** — metric chính | **0.597327402** |
| **ROUGE-L** | **0.586192885** |

Đây là hai giá trị trong `scores.json` đội nhận từ Codabench. File `submission.zip` đã nộp có SHA-256 `d697c48e4ce48410ad4695291c98891f9f872a88cf0f6d1dfc576d8dbf86496b`. Bản ZIP chứa 1.918 câu trả lời và đã qua bước kiểm tra format, số lượng ID ở local. Repo chưa có bản export Codabench lưu lâu dài để đối chiếu độc lập điểm với lần nộp này. Chúng tôi không công bố thứ hạng hoặc giải thưởng khi chưa có bằng chứng tương ứng.

## Vì sao chọn E45?

Ở E45, dữ liệu fine-tune dùng cùng cách lấy `parent context` và cùng kiểu prompt với lúc inference. Mục tiêu là giảm độ lệch giữa ngữ cảnh model thấy khi học và ngữ cảnh nó gặp khi trả lời câu hỏi. Adapter được train trên **5.636 record chính thức**, trong **705 optimizer step**, với `assistant-only loss` và giới hạn **8.192 token** cho cả prompt lẫn câu trả lời. Cấu hình đã đóng băng ghi nhận tổng **3.668.660.224 tham số** cho các model trong hệ thống, dưới ngưỡng *nhỏ hơn 4 tỷ* của cuộc thi.

[Kiến trúc chi tiết](docs/architecture.md) mô tả offline index và luồng inference. [Nhật ký thí nghiệm](docs/experiment-ledger.md) giải thích các quyết định từ E00/E02 đến E45, cùng giới hạn khi so sánh kết quả giữa các lần chạy.

## Bản nộp thực tế có một ngoại lệ

Bốn shard chạy private test trên Kaggle và tạo đủ 1.918 câu trả lời. Theo chính sách inference đã định, câu bị cắt vì chạm giới hạn 1.024 token sẽ được sinh lại từ input gốc với giới hạn 1.536 token. Khi đến deadline, **143 lượt sinh lại chưa xong**. Bản nộp dùng câu trả lời lượt đầu đã lưu trong checkpoint cho 143 trường hợp đó; các lượt sinh lại đã hoàn tất vẫn được giữ nguyên. Điểm private ở trên thuộc **đúng bản nộp này**, không phải một lần chạy mà mọi lượt sinh lại đều hoàn tất. Xem [báo cáo bản nộp](docs/submitted-run.md) và [giới hạn của kết quả](docs/limitations.md).

## Repo có gì?

| Thư mục | Nội dung |
|---|---|
| `src/uit_dsc_fixed_rag/` | Code retrieval, dựng context, chuẩn bị dữ liệu train, train và inference; có cả các module thí nghiệm trước E45 |
| `src/e45_private_*.py` | Chia shard, chạy hai T4 và lưu checkpoint theo từng câu |
| `scripts/` | Các entry point dùng để chuẩn bị, train, chạy private và ghép bản nộp sát deadline |
| `notebooks/` | Notebook Account A cuối cùng và bốn notebook private có checkpoint |
| `configs/` | Cấu hình E45 đã đóng băng |
| `tests/` | Test CPU/static cho format, hash, checkpoint và một số hợp đồng hành vi |
| `docs/` | Thiết kế, nội quy cuộc thi, lịch sử quyết định, hash artifact và cách tái lập |

Các module thí nghiệm cũ giúp lần theo quá trình chọn E45; chúng không đồng nghĩa với việc tất cả đều nằm trong runtime của bản nộp. Đây là repo đã tuyển chọn từ quá trình phát triển, không phải bản copy toàn bộ workspace.

## Chạy kiểm tra source

Không cần dữ liệu private để chạy các test nhỏ:

```bash
python -m compileall -q src scripts
python -m pytest -q tests/test_e45_static_prepare_jsonl.py tests/test_private_checkpoint_resume.py
```

Muốn chạy lại pipeline đầy đủ cần dữ liệu chính thức, model đúng revision, E00/E02 index, adapter E45 và GPU. Các file đó **không có trong Git**. [Hướng dẫn tái lập](docs/reproducibility.md) ghi rõ notebook nào đã dùng, runtime cần khớp và các bước kiểm tra artifact. [Danh sách SHA-256](docs/artifact-hashes.md) giúp nhận diện đúng đầu vào mà không phát tán dữ liệu.

## Ranh giới công bố

Repo không chứa dữ liệu train/public/private của BTC, câu trả lời private, `submission.zip`, index, checkpoint hay trọng số model. Hệ thống chỉ dùng dữ liệu chính thức; không dùng synthetic QA, corpus pháp luật ngoài cuộc thi hoặc model API. Format Codabench đã dùng là ZIP chỉ chứa một file `submission.json` UTF-8, với mỗi ID ánh xạ tới `{"answer": "..."}`. [Nội quy và hợp đồng chấm điểm](docs/competition-rules.md) ghi lại nguồn đã kiểm tra và cách E45 tuân thủ.

[Ghi nhận nguồn và quyền sử dụng](docs/attribution.md) phân biệt source của đội với model, dữ liệu và scorer từ bên ngoài. Repo chưa gắn giấy phép open-source cho code khi quyền của các thành viên và nghĩa vụ với upstream chưa được xác nhận.
