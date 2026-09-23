#!/usr/bin/env python3
"""Chạy hoặc đóng gói một private shard E45 đã được xác thực."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
# Bản public chỉ giữ các module E45 và private shard cần thiết trong src/.

from e45_private_shards import (  # noqa: E402
    PRIVATE_EXECUTION_STATUS_SCHEMA,
    compute_json_sha256,
    package_shard,
    run_private_shard,
)
from uit_dsc_fixed_rag.corpus import file_sha256  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("run", "package"))
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--candidate-bin", type=Path, required=True)
    parser.add_argument("--private-questions", type=Path, required=True)
    parser.add_argument("--e00", type=Path, required=True)
    parser.add_argument("--e02", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--checkpoint-archive", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--wall-clock-seconds", type=int)
    args = parser.parse_args()
    if args.phase == "run":
        result = run_private_shard(
            admission_path=args.admission, candidate_archive=args.candidate_bin,
            private_path=args.private_questions, shard_index=args.shard_index,
            shard_count=args.shard_count, e00_dir=args.e00, dense_dir=args.e02,
            tokenizer_path=args.tokenizer, output_dir=args.output, config_path=args.config,
            checkpoint_archive=args.checkpoint_archive,
            resume_checkpoint=args.resume_checkpoint,
            wall_clock_seconds=args.wall_clock_seconds,
        )
    else:
        status_path = args.output / "execution-status.json"
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit("package requires a valid completed or checkpointed execution-status.json") from exc
        body = {key: value for key, value in status.items() if key != "execution_status_sha256"}
        if (
            status.get("schema") != PRIVATE_EXECUTION_STATUS_SCHEMA
            or status.get("execution_status_sha256") != compute_json_sha256(body)
            or status.get("status") not in {"COMPLETE", "CHECKPOINTED"}
            or status.get("checkpoint_archive_sha256") != file_sha256(args.checkpoint_archive)
        ):
            raise SystemExit("Private execution status or checkpoint identity is invalid")
        if status["status"] == "CHECKPOINTED":
            sidecar = Path(str(args.checkpoint_archive) + ".sha256")
            if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").split()[0].lower() != status["checkpoint_archive_sha256"]:
                raise SystemExit("Checkpointed execution requires its matching sidecar")
            result = {
                "status": "CHECKPOINTED",
                "checkpoint": str(args.checkpoint_archive),
                "sha256": status["checkpoint_archive_sha256"],
                "completed_first_pass": status["completed_first_pass"],
                "total": status["total"],
            }
        else:
            if args.archive is None:
                raise SystemExit("complete private execution requires --archive")
            archive, digest = package_shard(output_dir=args.output, archive_path=args.archive)
            result = {"status": "COMPLETE", "archive": str(archive), "sha256": digest}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
