#!/usr/bin/env python3
"""Tái tạo câu hỏi holdout và chuẩn bị context mà không đọc đáp án."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uit_dsc_fixed_rag.e02_answer import _atomic_json
from uit_dsc_fixed_rag.e45_holdout_contexts import (
    prepare_holdout_contexts,
    select_group_safe_holdout,
)
from uit_dsc_fixed_rag.e45_parent_training import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOG = logging.getLogger("prepare_holdout")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", type=Path, default=PROJECT_ROOT / "configs/e45-inference-aligned-parent-lora-v1.json")
    parser.add_argument("--official-train", type=Path, required=True, help="Path to official train.json")
    parser.add_argument("--official-warmup", type=Path, required=True, help="Path to official warmup.json")
    parser.add_argument("--official-public", type=Path, required=True, help="Path to official public-official.json")
    parser.add_argument("--splits-train", type=Path, required=True, help="Path to frozen split train.json")
    parser.add_argument("--splits-dev", type=Path, required=True, help="Path to frozen split dev.json")
    parser.add_argument("--e00-dir", type=Path, required=True, help="Path to E00 dataset directory")
    parser.add_argument("--e02-dir", type=Path, required=True, help="Path to E02 dataset directory")
    parser.add_argument("--tokenizer-path", type=Path, required=True, help="Path to Vi-Qwen tokenizer")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/holdout", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    LOG.info("Starting holdout context preparation...")
    config = load_config(args.config_path)

    # 1. Tái tạo 200 câu hỏi holdout.
    questions, manifest = select_group_safe_holdout(
        official_train_path=args.official_train,
        official_warmup_path=args.official_warmup,
        official_public_path=args.official_public,
        splits_train_path=args.splits_train,
        splits_dev_path=args.splits_dev,
        config=config,
    )
    LOG.info("Reproduced %d holdout questions (Sample-ID SHA256: %s)", len(questions), manifest["sample_ids_sha256"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    questions_file = args.output_dir / "questions.json"
    _atomic_json(questions_file, {q["question_id"]: {"question": q["question"]} for q in questions})

    # 2. Retrieval và mở rộng context theo luồng P00/P01 đã chốt.
    contexts_file, context_manifest = prepare_holdout_contexts(
        questions=questions,
        e00_dir=args.e00_dir,
        dense_dir=args.e02_dir,
        tokenizer_path=args.tokenizer_path,
        output_dir=args.output_dir,
        config=config,
        device=args.device,
    )
    LOG.info("Holdout contexts written to %s (SHA256: %s)", contexts_file, context_manifest["contexts_jsonl_sha256"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
