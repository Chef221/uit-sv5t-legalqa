#!/usr/bin/env python3
"""Ghép kết quả hai Account E45 và chấm điểm ở local bằng scorer chính thức."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Cho phép import module từ thư mục gốc và src/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uit_dsc_fixed_rag.e45_finalize import merge_and_evaluate_e45
from uit_dsc_fixed_rag.e45_parent_training import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOG = logging.getLogger("merge_e45")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--account-a-bin",
        type=Path,
        required=True,
        help="Path to Account A output E45_ACCOUNT_A_CANDIDATE.bin",
    )
    parser.add_argument(
        "--account-b-bin",
        type=Path,
        required=True,
        help="Path to Account B output E45_ACCOUNT_B_CONTROL.bin",
    )
    parser.add_argument(
        "--references-file",
        type=Path,
        required=True,
        help="Path to sealed LOCAL_ONLY_HOLDOUT_REFERENCES.json",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=PROJECT_ROOT / "configs/e45-inference-aligned-parent-lora-v1.json",
        help="Path to E45 config JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/evaluation-results",
        help="Directory to write decision report and extracted files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    LOG.info("Starting E45 two-account merge and official evaluation...")
    config = load_config(args.config_path)

    results = merge_and_evaluate_e45(
        account_a_bin=args.account_a_bin,
        account_b_bin=args.account_b_bin,
        sealed_references_path=args.references_file,
        config=config,
        output_dir=args.output_dir,
    )

    metrics = results["metrics"]
    print("\n" + "=" * 65)
    print("E45 TWO-ACCOUNT FINAL VERDICT:", results["verdict"])
    print("=" * 65)
    print(f"Mean Paired METEOR Delta: {metrics['mean_paired_meteor_delta']:+.4f} (Gate >= +0.010)")
    print(f"Bootstrap 95% CI: [{metrics['bootstrap_95_ci'][0]:+.4f}, {metrics['bootstrap_95_ci'][1]:+.4f}] (Gate lower > 0)")
    print(f"Mean Paired ROUGE-L Delta: {metrics['mean_paired_rougel_delta']:+.4f} (Gate >= 0)")
    print(f"Candidate Length Finishes: {results['length_finishes']['candidate_final_length_finishes']} vs Control: {results['length_finishes']['control_final_length_finishes']}")
    print(f"Improved: {metrics['improved_count']} | Tied: {metrics['tied_count']} | Worsened: {metrics['worsened_count']}")
    print("=" * 65 + "\n")

    return 0 if results["verdict"] == "E45_PASSES_HELDOUT_GATE_PENDING_LEAD_REVIEW" else 2


if __name__ == "__main__":
    sys.exit(main())
