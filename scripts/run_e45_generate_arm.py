#!/usr/bin/env python3
"""Chạy một nhánh E45 bằng lịch hai replica P01 đã chốt."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uit_dsc_fixed_rag.e45_paired_generation import E45GeneratorWorker
from uit_dsc_fixed_rag.e45_parent_training import E45Error, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOG = logging.getLogger("generate_arm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", type=str, required=True, choices=["candidate", "control"])
    parser.add_argument("--contexts-path", type=Path, required=True, help="Path to holdout-contexts.jsonl")
    parser.add_argument("--base-model-path", type=str, required=True, help="HF model ID or local path to base model")
    parser.add_argument("--adapter-path", type=Path, required=True, help="Path to adapter checkpoint")
    parser.add_argument("--config-path", type=Path, default=PROJECT_ROOT / "configs/e45-inference-aligned-parent-lora-v1.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    LOG.info("Starting %s generation arm...", args.arm)
    config = load_config(args.config_path)

    if not args.contexts_path.is_file():
        raise E45Error(f"Prepared context artifact is missing: {args.contexts_path}")
    contexts = [json.loads(line) for line in args.contexts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    worker = E45GeneratorWorker(
        base_model_path=args.base_model_path,
        tokenizer_path=args.tokenizer_path,
        adapter_path=args.adapter_path,
        config=config,
    )
    raw_path, clean_path, report = worker.execute_p01_generation(
        contexts=contexts,
        output_dir=args.output_dir,
        arm_name=args.arm,
    )
    LOG.info("[%s] completed: raw=%s clean=%s report=%s", args.arm, raw_path, clean_path, report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
