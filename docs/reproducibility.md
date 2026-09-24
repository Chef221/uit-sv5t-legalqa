# Tái lập hệ thống từ source

Repo này **chỉ có source**. Muốn chạy lại đầy đủ cần lấy dữ liệu Task 2 từ kênh chính thức của BTC, tải model đúng revision trong `configs/e45-inference-aligned-parent-lora-v1.json`, rồi cung cấp SQLite BM25 index, dense vector index và LoRA adapter theo [hash đã ghi](artifact-hashes.md). Không đưa các file đó vào Git. Tên config/notebook vẫn mang mã thí nghiệm để giữ khớp với artifact đã chạy; [bảng đối chiếu](experiment-ledger.md#mã-thí-nghiệm-còn-trong-source) giải thích vai trò kỹ thuật của chúng.

## Đường chạy của bản nộp

1. `scripts/run_e45_static_prepare.py` kiểm tra config đã chốt và materialize 5.636 record train từ dữ liệu chính thức.
2. `scripts/run_e45_train_kaggle.py` được gọi từ `notebooks/E45-ACCOUNT-A-RESUME-AND-DIRECT-PRIVATE-PACKAGE-T4X2.ipynb`. Account A dùng hai T4 với DDP world size bằng hai, hoàn tất candidate sau 705 optimizer step.
3. Bốn notebook `notebooks/E45-PRIVATE-RESUMABLE-SHARD-*-OF-4-T4X2.ipynb` chuẩn bị context private và sinh answer theo shard. Mỗi notebook kiểm tra hash của system archive, release token và các input trước khi load model. Ở lần chạy mới, giữ `RUN_MODE = "fresh"` trong cell chọn input và kiểm tra log `FRESH_RUN_CONFIRMED`.
4. `scripts/merge_e45_emergency_checkpoints.py` ghép **bản nộp thực tế sát deadline**. Script giữ các lượt sinh lại đã xong và dùng answer lượt đầu cho 143 lượt sinh lại còn dang dở. Merger cho trường hợp mọi lượt sinh lại đều xong là một chính sách khác; không dùng nó để nhận điểm private đã công bố.

Nếu một shard trả `CHECKPOINTED`, lấy checkpoint `.bin` cùng sidecar `.sha256` từ chính lần chạy đó, attach vào session mới của **cùng shard**, đổi `RUN_MODE = "resume"` rồi chạy từ đầu notebook. Cell chọn input phải in `RESUME_CONFIRMED` và `saved_progress` trước khi tải model. Nếu thiếu checkpoint hoặc sidecar, notebook dừng; không tự chạy mới. Context được dựng lại và đối chiếu identity trước khi các câu đã lưu được bỏ qua. Không dùng checkpoint của lần nộp cũ để resume một lần chạy mới vì code identity có thể khác.

Notebook trong repo dựa trên notebook launch đã dùng khi thi; sau cuộc thi, bốn notebook private được bổ sung chốt `fresh/resume` để tránh âm thầm chạy lại từ đầu. Chúng cần các system archive có hash đúng, được giữ ngoài Git. `scripts/build_e45_private_shards.py` nhận đường dẫn R3 archive qua biến môi trường `E45_R3_SYSTEM_BIN` và ghi output vào `artifacts/` (Git bỏ qua thư mục này). Bản source public đã bỏ đường dẫn máy cá nhân và dùng layout repo mới, nên không byte-identical với system archive đã chạy trên Kaggle. Khi truy nguồn lần chạy thật, hãy dùng hash của archive gốc.

Runtime Python package đã pin trong mục `runtime` của config. Khi chạy trên GPU khác, vẫn phải kiểm tra driver, CUDA, model snapshot và artifact thực tế. Test CPU không thay cho một lần chạy GPU đầy đủ.

## Kiểm tra nhanh, không cần dữ liệu private

Tạo Python environment có `pytest`, rồi chạy:

```bash
python -m compileall -q src scripts
python -m pytest -q tests/test_e45_static_prepare_jsonl.py tests/test_private_checkpoint_resume.py
```

Các test cần framework train, model lớn hoặc dữ liệu BTC không nằm trong lệnh nhanh này. Resolver kiểm tra input bằng SHA-256; thiếu file hoặc gặp nhiều bản khớp đều dừng. Private inference và bước ghép cuối không nhận reference answer làm input.
