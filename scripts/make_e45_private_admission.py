#!/usr/bin/env python3
"""Tạo release token local cho private E45, gắn với candidate đã hoàn tất.

Chế độ thông thường yêu cầu vượt qua holdout. Chế độ direct-release được dùng
theo quyết định của đội: chạy private ngay sau khi Account A hoàn tất.
Cả hai chế độ đều xác minh archive candidate và bộ câu hỏi private chính thức.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PRIVATE_ROOT = Path(__file__).resolve().parents[1]
E45_ROOT = PRIVATE_ROOT
sys.path.insert(0, str(PRIVATE_ROOT / "src"))

from e45_private_shards import (  # noqa: E402
    ADMISSION_SCHEMA,
    DIRECT_RELEASE_AUTHORIZATION,
    DIRECT_RELEASE_SCHEMA,
    compute_json_sha256,
    load_private_questions,
    materialize_candidate_adapter,
)
from uit_dsc_fixed_rag.corpus import file_sha256  # noqa: E402
from uit_dsc_fixed_rag.e45_parent_training import load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--decision-json", type=Path)
    mode.add_argument("--direct-candidate", action="store_true")
    parser.add_argument("--candidate-bin", type=Path, required=True)
    parser.add_argument("--private-questions", type=Path, required=True)
    parser.add_argument("--e45-config", type=Path, default=E45_ROOT / "configs/e45-inference-aligned-parent-lora-v1.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    decision: dict[str, object] | None = None
    if args.decision_json is not None:
        decision = json.loads(args.decision_json.read_text(encoding="utf-8"))
        if decision.get("verdict") != "E45_PASSES_HELDOUT_GATE_PENDING_LEAD_REVIEW" or decision.get("zero_violations") is not True:
            raise SystemExit("Private admission blocked: the E45 heldout decision is not a zero-violation pass.")
    config = load_config(args.e45_config)
    _, _, private_identity = load_private_questions(args.private_questions)
    candidate_sha = file_sha256(args.candidate_bin)
    # Trình duyệt có thể đổi đuôi .bin thành .zip khi tải xuống.
    # Tìm đúng một sidecar theo hash, không dựa vào basename.
    sidecars = []
    for sidecar in args.candidate_bin.parent.glob("*.sha256"):
        try:
            fields = sidecar.read_text(encoding="utf-8").strip().split()
        except OSError as exc:
            raise SystemExit(f"Cannot read candidate SHA-256 sidecar: {sidecar}") from exc
        if fields and fields[0].lower() == candidate_sha:
            sidecars.append(sidecar)
    if len(sidecars) != 1:
        raise SystemExit("Candidate archive requires exactly one matching SHA-256 sidecar in its directory.")

    provisional = {
        "candidate_archive_sha256": candidate_sha,
        "candidate_adapter_sha256": "", "candidate_adapter_config_sha256": "", "candidate_complete_sha256": "",
    }
    from tempfile import TemporaryDirectory
    with TemporaryDirectory(prefix="e45_admission_") as temporary:
        # materialize_candidate_adapter kiểm tra manifest và 705 step đã hoàn tất.
        # Lấy hash adapter từ nội dung giải nén.
        candidate_root = Path(temporary) / "adapter"
        provisional.update({
            "candidate_adapter_sha256": "0" * 64,
            "candidate_adapter_config_sha256": "0" * 64,
            "candidate_complete_sha256": "0" * 64,
        })
        # Hàm kiểm tra cần hash kỳ vọng thực tế. Giải nén an toàn vào thư mục
        # tạm để lấy các giá trị đó trước.
        from uit_dsc_fixed_rag.e45_checkpoint import safe_extract_archive
        extracted = Path(temporary) / "candidate"
        safe_extract_archive(args.candidate_bin, extracted)
        complete = json.loads((extracted / "complete.json").read_text(encoding="utf-8"))
        training = config.section("training")
        model = config.section("model_inventory")
        expected_complete = {
            "experiment_id": config.experiment_id,
            "config_sha256": config.sha256,
            "global_step": training["expected_optimizer_steps"],
            "expected_total_steps": training["expected_optimizer_steps"],
            "base_model": model["base_model"],
            "base_revision": model["base_revision"],
            "lora_rank": training["lora_rank"],
            "lora_alpha": training["lora_alpha"],
            "lora_dropout": training["lora_dropout"],
            "target_modules": training["target_modules"],
        }
        if any(complete.get(key) != expected for key, expected in expected_complete.items()):
            raise SystemExit("Candidate complete.json does not match the frozen E45 training identity.")
        provisional["candidate_adapter_sha256"] = file_sha256(extracted / "adapter" / "adapter_model.safetensors")
        provisional["candidate_adapter_config_sha256"] = file_sha256(extracted / "adapter" / "adapter_config.json")
        provisional["candidate_complete_sha256"] = file_sha256(extracted / "complete.json")
        admission_for_validation = {"candidate_archive_sha256": candidate_sha, **provisional}
        materialize_candidate_adapter(candidate_archive=args.candidate_bin, destination=candidate_root, admission=admission_for_validation)

    payload = {
        "schema": ADMISSION_SCHEMA if decision is not None else DIRECT_RELEASE_SCHEMA,
        "experiment_id": config.experiment_id,
        "config_sha256": config.sha256,
        "candidate_archive_sha256": candidate_sha,
        "candidate_adapter_sha256": provisional["candidate_adapter_sha256"],
        "candidate_adapter_config_sha256": provisional["candidate_adapter_config_sha256"],
        "candidate_complete_sha256": provisional["candidate_complete_sha256"],
        "private_sha256": private_identity["private_sha256"],
        "private_sample_ids_sha256": private_identity["sample_ids_sha256"],
        "private_sample_size": private_identity["sample_size"],
        "issued_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if decision is not None:
        payload["heldout_verdict"] = decision["verdict"]
        payload["heldout_decision_sha256"] = file_sha256(args.decision_json)
    else:
        payload["authorization"] = DIRECT_RELEASE_AUTHORIZATION
    payload["admission_sha256"] = compute_json_sha256(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ADMITTED" if decision is not None else "DIRECT_RELEASED", "output": str(args.output), "admission_sha256": payload["admission_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
