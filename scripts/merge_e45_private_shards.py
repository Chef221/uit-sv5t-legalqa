#!/usr/bin/env python3
"""Ghép bốn bundle private shard E45 đã kiểm tra thành submission.zip."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
# Bản public chỉ giữ các module E45 và private shard cần thiết trong src/.

from e45_private_shards import merge_private_shards  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--private-questions", type=Path, required=True)
    parser.add_argument("--shard-bin", type=Path, required=True, action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = merge_private_shards(
        admission_path=args.admission, private_path=args.private_questions,
        shard_archives=args.shard_bin, output_dir=args.output,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
