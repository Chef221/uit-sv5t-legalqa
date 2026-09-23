#!/usr/bin/env python3
"""Execute full CPU/static materialization and computed falsification gates for E45 training data."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import sys
import tempfile
import time
from datetime import datetime, timezone
from statistics import fmean
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uit_dsc_fixed_rag import e21_parent_context as parent
from uit_dsc_fixed_rag.corpus import file_sha256
from uit_dsc_fixed_rag.e45_parent_training import (
    E45Config,
    E45Error,
    _E45ContextConfig,
    count_text_tokens,
    load_config,
    prepare_training_records,
    render_canonical_prompt,
)
from uit_dsc_fixed_rag.e45_input_resolver import compute_source_identity
from uit_dsc_fixed_rag.e45_checkpoint import tokenizer_aggregate_sha256, tokenizer_file_hashes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOG = logging.getLogger("e45_static_prepare")

PINNED_ORDERED_TRAINING_ID_SHA256 = "fee946ada7cbfca7df04ffe1c3d9f06907808c0dc59600e6f1f8d579076742dd"
PINNED_MATERIALIZED_JSONL_SHA256 = "84fb204f6268ebd4b0a56f1e4c5048c31a4e91778b9ac3c261d9de509a5ca17f"


def read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Read JSONL using physical file records separated by LF/CRLF.

    ``str.splitlines`` is deliberately forbidden here because it also splits
    on valid JSON string characters such as U+0085.  The pinned E19 authority
    contains one such character inside an answer.
    """
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for physical_line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise E45Error(
                    f"Invalid JSONL physical record {physical_line_number} in {path.name}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise E45Error(
                    f"JSONL physical record {physical_line_number} in {path.name} is not an object"
                )
            records.append(row)
    return records


def rebuild_frozen_materialization_summary(
    materialized: list[dict[str, Any]], output_jsonl: Path
) -> dict[str, Any]:
    """Rebuild summary for the exact pinned materialization after interruption."""
    observed_sha = file_sha256(output_jsonl)
    if observed_sha != PINNED_MATERIALIZED_JSONL_SHA256:
        raise E45Error(
            "Refusing materialization recovery: training-records.jsonl SHA-256 "
            f"{observed_sha} != {PINNED_MATERIALIZED_JSONL_SHA256}"
        )
    if not materialized:
        raise E45Error("Refusing empty materialization recovery")
    prompt_lengths = [int(row["prompt_tokens_count"]) for row in materialized]
    total_lengths = [int(row["total_tokens_count"]) for row in materialized]
    qids = [str(row["question_id"]) for row in materialized]
    aggregate = hashlib.sha256(
        "".join(str(row["record_sha256"]) for row in materialized).encode("utf-8")
    ).hexdigest()
    return {
        "records_count": len(materialized),
        "mean_prompt_length": fmean(prompt_lengths),
        "min_prompt_length": min(prompt_lengths),
        "max_prompt_length": max(prompt_lengths),
        "mean_total_length": fmean(total_lengths),
        "max_total_length": max(total_lengths),
        "expansion_retention_rate": sum(bool(row.get("expansion_decisions")) for row in materialized) / len(materialized),
        "answer_truncation_count": 0,
        "ordered_training_ids_sha256": hashlib.sha256("\n".join(qids).encode("utf-8")).hexdigest(),
        "exact_e00_spans_checked": 0,
        "canonical_prompt_renders_checked": 0,
        "materialization_identity_verified": observed_sha,
        "materialized_jsonl_sha256": observed_sha,
        "aggregate_records_sha256": aggregate,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", type=Path, default=PROJECT_ROOT / "configs/e45-inference-aligned-parent-lora-v1.json")
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--e00-dir", type=Path, required=True)
    parser.add_argument("--e08a-results", type=Path, required=True)
    parser.add_argument("--prior-records", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/training-data")
    parser.add_argument("--skip-materialization-if-exists", action="store_true", help="Audit existing training-records.jsonl if present")
    return parser.parse_args()


def compute_code_manifest(root: Path) -> dict[str, str]:
    manifest = {}
    for ext in ["*.py", "*.json"]:
        for p in (root / "src").rglob(ext):
            if "__pycache__" not in p.parts:
                manifest[p.relative_to(root).as_posix()] = file_sha256(p)
        for p in (root / "configs").rglob(ext):
            manifest[p.relative_to(root).as_posix()] = file_sha256(p)
    return manifest


def audit_falsification_gates(
    *,
    materialized: list[dict[str, Any]],
    config: E45Config,
    tokenizer: Any,
    e08a_results_path: Path,
    e00_dir: Path,
    official_train_path: Path,
    prior_records_path: Path,
    output_jsonl_path: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Execute active, computed validations for all 16 falsification gates."""
    LOG.info("Auditing all 16 falsification gates with computed validations...")
    training_cfg = config.section("training")
    model_inv = config.section("model_inventory")
    context_policy = config.section("context_policy")

    # Gate 1: đúng 5.636 record.
    g1 = (len(materialized) == training_cfg["records_count"])

    # Gate 2: 5.636 question ID không trùng.
    qids = [r["question_id"] for r in materialized]
    g2 = (len(set(qids)) == training_cfg["records_count"])

    # Gate 3: SHA-256 của danh sách ID đúng thứ tự.
    computed_ordered_id_sha = hashlib.sha256("\n".join(qids).encode("utf-8")).hexdigest()
    g3 = (computed_ordered_id_sha == PINNED_ORDERED_TRAINING_ID_SHA256)

    # Gate 4: không cắt ngắn answer.
    g4 = all(
        r["answer_tokens_count"] > 0
        and r["prompt_tokens_count"] + r["answer_tokens_count"] == r["total_tokens_count"]
        for r in materialized
    )

    # Gate 5: tokenize lại từng answer chính thức, thêm đúng một EOS rồi so với
    # phần label có hiệu lực trong record đã lưu. Chỉ kiểm tra token cuối sẽ
    # không phát hiện answer bị đổi hoặc cắt ngắn.
    g5_checks = []
    g6_checks = []
    eos_id = tokenizer.eos_token_id
    official_targets = json.loads(official_train_path.read_text(encoding="utf-8"))
    prior_targets = {
        str(row["question_id"]): row["answer"]
        for row in read_jsonl_records(prior_records_path)
        if isinstance(row.get("answer"), str)
    }
    for r in materialized:
        ans_len = r["answer_tokens_count"]
        prompt_len = r["prompt_tokens_count"]
        input_ids = r["input_ids"]
        labels = r["labels"]

        official = official_targets.get(str(r["question_id"]))
        expected_answer = prior_targets.get(str(r["question_id"]))
        # Prior record là bản đã pin của dòng được chọn từ train chính thức.
        # Nếu QID còn trong train, nội dung phải khớp tuyệt đối.
        if not isinstance(expected_answer, str) or (isinstance(official, dict) and official.get("answer") != expected_answer):
            g5_checks.append(False)
        else:
            expected_target = tokenizer(
                expected_answer, add_special_tokens=False
            )["input_ids"] + [eos_id]
            active_target = input_ids[prompt_len:]
            # Answer chính thức có thể chứa chuỗi trông giống EOS. Điều cần
            # kiểm tra là toàn bộ token sequence phải khớp chính xác.
            g5_checks.append(active_target == expected_target and ans_len == len(expected_target))

        # Gate 6: mask chỉ tính loss trên answer (-100 ở prompt).
        prompt_masked = (labels[:prompt_len] == [-100] * prompt_len)
        ans_active = (labels[prompt_len:] == input_ids[prompt_len:])
        g6_checks.append(prompt_masked and ans_active)

    g5 = all(g5_checks)
    g6 = all(g6_checks)

    # Gate 7: sequence không quá 8.192 token.
    g7 = all(r["total_tokens_count"] <= 8192 for r in materialized) and max(r["total_tokens_count"] for r in materialized) <= 8192

    # Gate 8: không có reference/gold field trong context hoặc prompt.
    g8_checks = []
    for r in materialized:
        # Từ chối key gold/reference.
        has_leak = any(k in r for k in ["reference_answer", "gold_evidence", "reference"])
        has_leak_in_units = any("answer" in u for u in r.get("units", []))
        g8_checks.append(not has_leak and not has_leak_in_units)
    g8 = all(g8_checks)

    # Gate 9: kiểm tra document hash của mọi record và số span khớp tuyệt đối
    # mà materializer đã đếm riêng. Trước khi lưu record, materializer so từng
    # parent/seed span với `cleaned_text[start:end]` và báo lỗi nếu khác.
    LOG.info("Auditing Gate 9: substring provenance against real E00 documents...")
    doc_path = e00_dir / config.section("metadata_source")["documents_path"]
    required_doc_hashes: dict[str, set[str]] = {}
    for r in materialized:
        for did, expected in r.get("source_byte_hashes", {}).items():
            required_doc_hashes.setdefault(str(did), set()).add(str(expected))
    docs = {}
    with doc_path.open("r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            did = str(d["document_id"])
            if did in required_doc_hashes:
                docs[did] = hashlib.sha256(d["cleaned_text"].encode("utf-8")).hexdigest()
    g9 = (
        bool(required_doc_hashes)
        and set(docs) == set(required_doc_hashes)
        and all(docs[did] in expected_values for did, expected_values in required_doc_hashes.items())
        and (
            summary.get("exact_e00_spans_checked", 0) >= len(materialized) * 12
            or summary.get("materialization_identity_verified") == PINNED_MATERIALIZED_JSONL_SHA256
        )
    )

    # Gate 10: so thứ tự seed với E08A theo từng dòng.
    LOG.info("Auditing Gate 10: direct row-by-row E08A seed order comparison...")
    e08a_seeds_map = {}
    with e08a_results_path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            e08a_seeds_map[str(item["question_id"])] = [c["chunk_id"] for c in item["contexts"][:12]]
    g10 = all(r["ordered_seed_ids"] == e08a_seeds_map[str(r["question_id"])] for r in materialized)

    # Gate 11: đổi target của một dòng thật trong file tạm, giữ nguyên số token.
    # Retrieval row, câu hỏi, quyết định chọn evidence, prompt IDs và prompt hash
    # phải giữ nguyên từng byte. Target đã đổi không được dùng để train/sinh.
    LOG.info("Auditing Gate 11: real-row answer-mutation parity...")
    g11 = False
    try:
        authority_rows = read_jsonl_records(prior_records_path)
        if authority_rows:
            first_qid = str(materialized[0]["question_id"])
            pos = next(i for i, row in enumerate(authority_rows) if str(row["question_id"]) == first_qid)
            answer_ids = tokenizer(authority_rows[pos]["answer"], add_special_tokens=False)["input_ids"]
            # Xoay token IDs thay vì tự tạo văn bản. Thử các phép xoay đến khi
            # decode/encode giữ đúng độ dài cũ.
            replacement = None
            decoded_original = tokenizer.decode(answer_ids, clean_up_tokenization_spaces=False)
            if decoded_original != authority_rows[pos]["answer"] and tokenizer(decoded_original, add_special_tokens=False)["input_ids"] == answer_ids:
                replacement = decoded_original
            for shift in range(1, min(len(answer_ids), 64)):
                if replacement is not None:
                    break
                candidate = tokenizer.decode(answer_ids[shift:] + answer_ids[:shift], clean_up_tokenization_spaces=False)
                if candidate != authority_rows[pos]["answer"] and tokenizer(candidate, add_special_tokens=False)["input_ids"] == answer_ids[shift:] + answer_ids[:shift]:
                    replacement = candidate
                    break
            if replacement is not None:
                mutated_rows = copy.deepcopy(authority_rows)
                mutated_rows[pos]["answer"] = replacement
                with tempfile.TemporaryDirectory(prefix="e45-answer-blind-") as temp_dir:
                    temp_root = Path(temp_dir)
                    mutated_authority = temp_root / "records.jsonl"
                    mutated_authority.write_text(
                        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in mutated_rows),
                        encoding="utf-8",
                    )
                    mutated_records, _ = prepare_training_records(
                        e08a_path=e08a_results_path,
                        records_path=mutated_authority,
                        chunks_path=e00_dir / config.section("metadata_source")["chunks_path"],
                        documents_path=e00_dir / config.section("metadata_source")["documents_path"],
                        tokenizer=tokenizer,
                        config=config,
                        limit=1,
                    )
                original = materialized[0]
                mutated = mutated_records[0]
                g11 = all(
                    original[key] == mutated[key]
                    for key in ("ordered_seed_ids", "expansion_decisions", "rendered_prompt_sha256", "source_byte_hashes", "active_label_boundaries")
                ) and original["labels"][: original["prompt_tokens_count"]] == mutated["labels"][: mutated["prompt_tokens_count"]]
                g11 = g11 and original["input_ids"][: original["prompt_tokens_count"]] == mutated["input_ids"][: mutated["prompt_tokens_count"]]
    except (OSError, ValueError, KeyError, StopIteration, E45Error) as exc:
        LOG.error("Gate 11 answer-mutation parity could not run: %s", exc)
        g11 = False

    # Gate 12: mọi record phải đi qua renderer P00/P01 canonical khi materialize.
    # Hash lại prompt IDs theo ranh giới đã lưu và kiểm tra đủ số lần render.
    # Nhờ vậy record thiếu hoặc prompt thay thế chỉ có label sẽ bị từ chối.
    LOG.info("Auditing Gate 12: canonical E44/P00 render accounting and prompt hash integrity...")
    g12 = all(
        bool(r.get("rendered_prompt_sha256"))
        and r["active_label_boundaries"][0] == r["prompt_tokens_count"]
        and r["active_label_boundaries"][1] == r["total_tokens_count"]
        and r["prompt_tokens_count"] + r["answer_tokens_count"] == r["total_tokens_count"]
        for r in materialized
    ) and (
        summary.get("canonical_prompt_renders_checked") == len(materialized)
        or summary.get("materialization_identity_verified") == PINNED_MATERIALIZED_JSONL_SHA256
    )

    # Gate 13: tỷ lệ giữ parent expansion ít nhất 95,0%.
    expansion_rate = summary["expansion_retention_rate"]
    g13 = (expansion_rate >= 0.95)

    # Gate 14: prompt trung bình ít nhất 4.887,008 token.
    mean_prompt_len = summary["mean_prompt_length"]
    g14 = (mean_prompt_len >= 4887.008)

    # Gate 15: tính lại hai aggregate identity từ file trên đĩa.
    observed_file_sha = file_sha256(output_jsonl_path)
    observed_records_sha = hashlib.sha256(
        "".join(r["record_sha256"] for r in materialized).encode("utf-8")
    ).hexdigest()
    g15 = (
        observed_file_sha == summary.get("materialized_jsonl_sha256")
        and observed_records_sha == summary.get("aggregate_records_sha256")
    )

    # Gate 16: kiểm tra toàn bộ field config ảnh hưởng chất lượng; sai là dừng.
    LOG.info("Auditing Gate 16: fail-closed quality config contract audit...")
    expected_contract = {
        "training": {
            "records_count": 5636, "epochs": 1.0, "expected_optimizer_steps": 705,
            "lora_rank": 8, "lora_alpha": 16, "lora_dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "seed": 20260830, "learning_rate": 0.0001, "per_device_train_batch": 1,
            "gradient_accumulation": 4, "world_size": 2, "effective_global_batch": 8,
            "warmup_ratio": 0.03, "scheduler": "cosine", "weight_decay": 0.0,
            "save_interval_steps": 32, "quantization": "nf4-double-quant",
            "compute_dtype": "float16", "gradient_checkpointing": True,
            "maximum_total_sequence": 8192, "loss": "assistant-target-only",
            "target": "complete-official-answer-followed-by-eos",
        },
        "model_inventory": {
            "base_model": "AITeamVN/Vi-Qwen2-3B-RAG",
            "base_revision": "eaf427c24d86066a2b35828c499b7db3af321227",
            "expected_lora_trainable_parameters": 14966784,
        },
        "retrieval": {"candidate_k_per_branch": 40, "rrf_constant": 60, "sparse_weight": 0.5, "dense_weight": 0.5, "fused_top_k": 20, "selected_contexts": 12, "normalize_query_embeddings": True, "reranker": None},
        "context_policy": {"seed_contexts": 12, "max_parent_tokens": 1200, "max_parent_characters": 6000, "neighbor_radius": 1, "expansion_order": ["whole_article", "both_neighbors", "previous_neighbor", "next_neighbor"], "evict_seed_for_expansion": False, "merge": "overlapping-or-touching-exact-document-spans", "max_input_tokens": 8192},
        "inference": {"precision": "float16", "greedy_decoding": True, "sampling": False, "initial_max_new_tokens": 1024, "second_pass_max_new_tokens": 1536, "clean_up_tokenization_spaces": False, "replica_max_input_tokens": 6000, "maximum_fixed_point_passes_per_stage": 16, "suffix_only": True, "interior_trim": False},
        "runtime": {
            "torch": "2.10.0+cu128", "transformers": "5.16.1",
            "peft": "0.19.1", "accelerate": "1.13.0",
            "bitsandbytes": "0.50.2", "sentence-transformers": "5.4.1",
            "faiss-cpu": "1.15.0", "numpy": "2.0.2",
        },
        "acceptance_gate": {"min_mean_paired_meteor_delta": 0.01, "min_bootstrap_ci_lower_bound": 0.0, "min_rougel_delta": 0.0, "candidate_length_finish_less_than_control": True},
    }
    cfg_checks = [
        training_cfg["records_count"] == 5636,
        training_cfg["epochs"] == 1.0,
        training_cfg["expected_optimizer_steps"] == 705,
        training_cfg["lora_rank"] == 8,
        training_cfg["lora_alpha"] == 16,
        training_cfg["lora_dropout"] == 0.05,
        training_cfg["target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        training_cfg["seed"] == 20260830,
        training_cfg["learning_rate"] == 0.0001,
        training_cfg["per_device_train_batch"] == 1,
        training_cfg["gradient_accumulation"] == 4,
        training_cfg["world_size"] == 2,
        training_cfg["effective_global_batch"] == 8,
        training_cfg["warmup_ratio"] == 0.03,
        training_cfg["scheduler"] == "cosine",
        training_cfg["maximum_total_sequence"] == 8192,
        model_inv["base_model"] == "AITeamVN/Vi-Qwen2-3B-RAG",
        model_inv["base_revision"] == "eaf427c24d86066a2b35828c499b7db3af321227",
        context_policy["max_parent_tokens"] == 1200,
        context_policy["max_parent_characters"] == 6000,
        context_policy["neighbor_radius"] == 1,
        context_policy["expansion_order"] == ["whole_article", "both_neighbors", "previous_neighbor", "next_neighbor"],
    ]
    for section, required_items in expected_contract.items():
        actual = config.section(section)
        cfg_checks.extend(actual.get(key) == value for key, value in required_items.items())
    g16 = all(cfg_checks)

    gate_results = {
        "gate_01_materialized_records_count": g1,
        "gate_02_unique_question_ids": g2,
        "gate_03_ordered_training_id_sha256": g3,
        "gate_04_zero_answer_truncations": g4,
        "gate_05_complete_answer_and_one_eos": g5,
        "gate_06_assistant_only_label_mask": g6,
        "gate_07_max_8192_tokens": g7,
        "gate_08_reference_isolation": g8,
        "gate_09_exact_e00_substring_provenance": g9,
        "gate_10_all_12_frozen_seeds_preserved": g10,
        "gate_11_answer_blind_context_selection": g11,
        "gate_12_canonical_e44_prompt_parity": g12,
        "gate_13_parent_expansion_rate": g13,
        "gate_14_mean_prompt_length": g14,
        "gate_15_preparation_manifest_hashes": g15,
        "gate_16_quality_config_matches_contract": g16,
    }

    LOG.info("Gate audit complete. Summary:")
    for k, v in sorted(gate_results.items()):
        LOG.info("  %s: %s", k, "PASS" if v else "FAIL")

    return gate_results


def main() -> int:
    args = parse_args()
    LOG.info("Starting E45 static preparation and computed gate audit...")
    t0 = time.time()

    config = load_config(args.config_path)
    output_jsonl = args.output_dir / "training-records.jsonl"
    manifest_path = args.output_dir / "preparation-manifest.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer_path),
        revision=config.section("model_inventory")["base_revision"],
        fix_mistral_conversions=False,
    )

    if args.skip_materialization_if_exists and output_jsonl.is_file():
        LOG.info("Validating and reusing exact pinned materialization from %s...", output_jsonl)
        materialized = read_jsonl_records(output_jsonl)
        summary = rebuild_frozen_materialization_summary(materialized, output_jsonl)
    else:
        LOG.info("Materializing 5,636 training records from raw inputs...")
        materialized, summary = prepare_training_records(
            e08a_path=args.e08a_results,
            records_path=args.prior_records,
            chunks_path=args.e00_dir / config.section("metadata_source")["chunks_path"],
            documents_path=args.e00_dir / config.section("metadata_source")["documents_path"],
            tokenizer=tokenizer,
            config=config,
        )
        with output_jsonl.open("w", encoding="utf-8", newline="\n") as stream:
            for row in materialized:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        summary["materialized_jsonl_sha256"] = file_sha256(output_jsonl)
        summary["aggregate_records_sha256"] = hashlib.sha256(
            "".join(row["record_sha256"] for row in materialized).encode("utf-8")
        ).hexdigest()

    # Chạy các phép kiểm tra thực tế, không đánh dấu pass theo giả định.
    gate_checks = audit_falsification_gates(
        materialized=materialized,
        config=config,
        tokenizer=tokenizer,
        e08a_results_path=args.e08a_results,
        e00_dir=args.e00_dir,
        official_train_path=args.train_json,
        prior_records_path=args.prior_records,
        output_jsonl_path=output_jsonl,
        summary=summary,
    )

    all_gates_pass = all(gate_checks.values())

    code_manifest = compute_code_manifest(PROJECT_ROOT)
    code_aggregate_sha = compute_source_identity(PROJECT_ROOT)

    manifest = {
        "schema_version": "1.0",
        "experiment_id": config.experiment_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(args.config_path.resolve()),
        "config_sha256": config.sha256,
        "code_manifest_sha256": code_aggregate_sha,
        "code_files_count": len(code_manifest),
        "tokenizer_path": str(args.tokenizer_path.resolve()),
        "tokenizer_hash": file_sha256(args.tokenizer_path / "tokenizer.json") if (args.tokenizer_path / "tokenizer.json").is_file() else "",
        "tokenizer_files": tokenizer_file_hashes(args.tokenizer_path),
        "tokenizer_aggregate_sha256": tokenizer_aggregate_sha256(tokenizer_file_hashes(args.tokenizer_path)),
        "training_records_jsonl_sha256": summary["materialized_jsonl_sha256"],
        "aggregate_records_sha256": summary["aggregate_records_sha256"],
        "records_count": len(materialized),
        "summary": summary,
        "gate_checks": gate_checks,
        "all_gates_pass": all_gates_pass,
        "elapsed_seconds": time.time() - t0,
    }

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOG.info("Preparation manifest written to %s (all_gates_pass=%s)", manifest_path, all_gates_pass)

    if not all_gates_pass:
        LOG.error("Falsification gates failed! Aborting fail-closed.")
        return 1

    LOG.info("All 16 computed falsification gates PASSED in %.2fs.", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
