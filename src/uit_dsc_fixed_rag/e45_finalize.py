"""Fail-closed E45 two-account merger and official-compatible heldout scorer."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .corpus import file_sha256
from .e45_checkpoint import safe_extract_archive
from .e45_input_resolver import compute_source_identity
from .e45_paired_generation import compute_json_sha256
from .evaluation.official import ensure_nltk_resources, nltk_meteor_score, official_rouge_tokens, rouge_l_fmeasure

PINNED_HOLDOUT_SAMPLE_IDS_SHA256 = "36c6faad7e7d52030ea98b00102281c707a798e33845a7ddb89f96473717d4d1"
PINNED_SEALED_REFERENCES_SHA256 = "a893ff4625dc09f44846d8d6c0d0c479148eac9378688071dff86d87f08ce575"
PINNED_E38_ADAPTER_SHA256 = "e8f55f088fe2c5951c336095a5682af3a53f62c8be186bd5bfbf34741d895222"


class FinalizeError(RuntimeError):
    """Raised before scoring whenever an E45 account output is unsafe or mixed."""


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FinalizeError(f"Required output member is missing: {path.name}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _validate_sidecar(archive: Path) -> None:
    if not archive.is_file():
        raise FinalizeError(f"Archive not found: {archive}")
    sidecar = Path(str(archive) + ".sha256")
    if not sidecar.is_file():
        raise FinalizeError(f"Archive sidecar not found: {sidecar}")
    declared = sidecar.read_text(encoding="utf-8").strip().split()[0].lower()
    actual = file_sha256(archive)
    if declared != actual:
        raise FinalizeError(f"Archive sidecar mismatch for {archive.name}")


def _validate_manifest(root: Path, label: str) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FinalizeError(f"{label} archive has no internal manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not manifest:
        raise FinalizeError(f"{label} archive manifest is invalid")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    expected = set(manifest) | {"manifest.json"}
    if actual != expected:
        raise FinalizeError(f"{label} archive members differ from its manifest")
    for rel, meta in manifest.items():
        path = root / rel
        if not path.is_file() or path.stat().st_size != meta.get("bytes") or file_sha256(path) != meta.get("sha256"):
            raise FinalizeError(f"{label} manifest mismatch: {rel}")
    return manifest


def _record_hash_valid(row: dict[str, Any]) -> bool:
    given = row.get("record_sha256")
    body = {key: value for key, value in row.items() if key != "record_sha256"}
    return isinstance(given, str) and compute_json_sha256(body) == given


def _no_reference_fields(value: Any, *, allowed_answer_key: str | None = None) -> None:
    """Reject reference-bearing schemas before sealed references are opened."""
    forbidden = {"reference", "reference_answer", "gold", "gold_answer", "gold_evidence", "target_answer", "official_answer"}
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = key.casefold()
            if lowered in forbidden or ("reference" in lowered and lowered != "reference_sha256"):
                raise FinalizeError(f"Reference-bearing field is prohibited in account output: {key}")
            if lowered == "answer" and key != allowed_answer_key:
                raise FinalizeError("Unqualified answer field is prohibited in account output")
            _no_reference_fields(item, allowed_answer_key=allowed_answer_key)
    elif isinstance(value, list):
        for item in value:
            _no_reference_fields(item, allowed_answer_key=allowed_answer_key)


def _runtime_identity(root: Path, label: str) -> dict[str, Any]:
    path = root / "runtime-identity.json"
    if not path.is_file():
        raise FinalizeError(f"{label} missing runtime-identity.json")
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {"experiment_id", "config_sha256", "code_sha256", "base_model", "base_revision", "tokenizer_sha256", "runtime_versions", "decoding_sha256", "cleanup_sha256", "context_artifact_sha256", "adapter_sha256", "identity_sha256"}
    if set(value) - (required | {"completed_at_utc", "arm_name"}) or not required.issubset(value):
        raise FinalizeError(f"{label} runtime identity has missing or unexpected fields")
    if compute_json_sha256({k: v for k, v in value.items() if k != "identity_sha256"}) != value["identity_sha256"]:
        raise FinalizeError(f"{label} runtime identity hash is invalid")
    return value


def _validate_contexts(rows: list[dict[str, Any]]) -> list[str]:
    if len(rows) != 200:
        raise FinalizeError(f"Prepared context count {len(rows)} != 200")
    qids: list[str] = []
    for index, row in enumerate(rows):
        required = {"sample_index", "question_id", "question", "question_sha256", "ordered_seed_ids", "evidence_sha256", "prompt", "prompt_sha256", "prompt_input_ids_sha256", "input_tokens", "units", "answers_used", "record_sha256"}
        if set(row) != required or row["sample_index"] != index or row["answers_used"] is not False:
            raise FinalizeError(f"Invalid prepared context schema at index {index}")
        if not _record_hash_valid(row) or hashlib.sha256(row["question"].encode("utf-8")).hexdigest() != row["question_sha256"]:
            raise FinalizeError(f"Prepared context identity mismatch at index {index}")
        _no_reference_fields(row)
        qids.append(row["question_id"])
    if hashlib.sha256("\n".join(qids).encode("utf-8")).hexdigest() != PINNED_HOLDOUT_SAMPLE_IDS_SHA256:
        raise FinalizeError("Prepared context IDs differ from frozen holdout")
    return qids


def _validate_arm(
    *, root: Path, label: str, contexts: list[dict[str, Any]], qids: list[str], expected_control: bool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    runtime = _runtime_identity(root, label)
    if runtime["context_artifact_sha256"] != file_sha256(root / "holdout-contexts.jsonl"):
        raise FinalizeError(f"{label} runtime identity is not bound to its exact context artifact")
    report = json.loads((root / "generation-report.json").read_text(encoding="utf-8"))
    report_required = {
        "schema_version", "experiment_id", "arm_name", "identity", "raw_records_sha256",
        "clean_records_sha256", "sample_size", "initial_length_finish_count", "final_length_finish_count", "second_pass_count",
        "worker_state_files", "completed_at_utc",
    }
    if set(report) != report_required or report["schema_version"] != "1.0" or report["sample_size"] != 200:
        raise FinalizeError(f"{label} generation report schema is invalid")
    runtime_generation_identity = {k: runtime[k] for k in ("experiment_id", "config_sha256", "base_model", "base_revision", "tokenizer_sha256", "adapter_sha256", "decoding_sha256", "cleanup_sha256")}
    if any(report.get("identity", {}).get(key) != value for key, value in runtime_generation_identity.items()):
        raise FinalizeError(f"{label} generation report identity does not bind runtime identity")
    report_identity = report["identity"]
    if compute_json_sha256({key: value for key, value in report_identity.items() if key != "identity_sha256"}) != report_identity.get("identity_sha256"):
        raise FinalizeError(f"{label} generation identity hash is invalid")
    if expected_control and runtime["adapter_sha256"] != PINNED_E38_ADAPTER_SHA256:
        raise FinalizeError("Control output does not use the pinned E38 adapter")

    raw, clean = _jsonl(root / "raw-records.jsonl"), _jsonl(root / "clean-records.jsonl")
    if report["raw_records_sha256"] != file_sha256(root / "raw-records.jsonl") or report["clean_records_sha256"] != file_sha256(root / "clean-records.jsonl"):
        raise FinalizeError(f"{label} generation report record-file hashes are invalid")
    if len(raw) != 200 or len(clean) != 200:
        raise FinalizeError(f"{label} raw/clean coverage is incomplete")
    raw_by_hash: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(raw):
        required = {"sample_index", "question_id", "context_record_sha256", "question_sha256", "evidence_sha256", "prompt_sha256", "prompt_input_ids_sha256", "experiment_id", "config_sha256", "base_model", "base_revision", "tokenizer_sha256", "adapter_sha256", "decoding_sha256", "cleanup_sha256", "identity_sha256", "worker_path", "raw_answer", "initial_finish_reason", "finish_reason", "first_pass_tokens", "second_pass_tokens", "fallback_reason", "record_sha256"}
        if set(row) != required or row["sample_index"] != index or row["question_id"] != qids[index] or not _record_hash_valid(row):
            raise FinalizeError(f"{label} raw record invalid at index {index}")
        prepared = contexts[index]
        for field, expected in {
            "context_record_sha256": prepared["record_sha256"], "question_sha256": prepared["question_sha256"],
            "evidence_sha256": prepared["evidence_sha256"], "prompt_sha256": prepared["prompt_sha256"],
            "prompt_input_ids_sha256": prepared["prompt_input_ids_sha256"],
        }.items():
            if row[field] != expected:
                raise FinalizeError(f"{label} raw record context binding failed at index {index}: {field}")
        for field in ("experiment_id", "config_sha256", "base_model", "base_revision", "tokenizer_sha256", "adapter_sha256", "decoding_sha256", "cleanup_sha256"):
            if row[field] != runtime[field]:
                raise FinalizeError(f"{label} raw record runtime binding failed at index {index}: {field}")
        _no_reference_fields(row, allowed_answer_key="raw_answer")
        raw_by_hash[row["record_sha256"]] = row
    for index, row in enumerate(clean):
        required = {"sample_index", "question_id", "raw_record_sha256", "cleanup_sha256", "clean_answer", "cleanup_report", "record_sha256"}
        if set(row) != required or row["sample_index"] != index or row["question_id"] != qids[index] or not _record_hash_valid(row):
            raise FinalizeError(f"{label} clean record invalid at index {index}")
        if row["raw_record_sha256"] not in raw_by_hash or row["cleanup_sha256"] != runtime["cleanup_sha256"]:
            raise FinalizeError(f"{label} clean record raw/cleanup binding failed at index {index}")
        _no_reference_fields(row, allowed_answer_key="clean_answer")

    state_dir = root / "worker-states"
    state_files = sorted(state_dir.glob("*-state.json")) if state_dir.is_dir() else []
    if len(state_files) < 2:
        raise FinalizeError(f"{label} missing durable replica worker states")
    covered: list[int] = []
    for path in state_files:
        state = json.loads(path.read_text(encoding="utf-8"))
        state_hash = state.get("state_sha256")
        if not state_hash or compute_json_sha256({k: v for k, v in state.items() if k != "state_sha256"}) != state_hash:
            raise FinalizeError(f"{label} worker-state hash mismatch: {path.name}")
        if state.get("identity") != report["identity"] or state.get("status") not in {"complete", "oom"}:
            raise FinalizeError(f"{label} worker-state identity/status mismatch: {path.name}")
        for index, record_sha in zip(state.get("completed", []), state.get("raw_record_sha256s", []), strict=True):
            if index < 0 or index >= 200 or raw[index]["record_sha256"] != record_sha:
                raise FinalizeError(f"{label} worker-state record binding mismatch: {path.name}")
        covered.extend(state.get("completed", []))
    if sorted(covered) != list(range(200)):
        raise FinalizeError(f"{label} worker states do not prove exact complete coverage")
    return raw, clean, runtime, report


def _official_rouge_tokens(text: str) -> list[str]:
    """Match BTC vendored rouge_score DefaultTokenizer: lowercase ASCII-only words."""
    return official_rouge_tokens(text)


def compute_official_rouge_l(reference: str, candidate: str) -> float:
    """Compute the documented BTC vendored ROUGE-L F1 tokenizer and formula."""
    return rouge_l_fmeasure(reference, candidate)


def compute_official_meteor(reference: str, candidate: str) -> float:
    """Use the BTC scorer's whitespace-token NLTK METEOR call."""
    return nltk_meteor_score(reference, candidate)


def _verify_official_scorer_runtime() -> dict[str, str]:
    """Require the pinned local NLTK scorer runtime before opening references."""
    observed = importlib.metadata.version("nltk")
    if observed != "3.7":
        raise FinalizeError(f"Local scorer requires NLTK 3.7, observed {observed}")
    ensure_nltk_resources(download=False)
    scorer_path = Path(__file__).parent / "evaluation" / "official.py"
    return {"nltk": observed, "official_scorer_sha256": file_sha256(scorer_path)}


def _bootstrap(deltas: list[float], seed: int = 20260830) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(deltas, dtype=np.float64)
    means = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def merge_and_evaluate_e45(*, account_a_bin: Path, account_b_bin: Path, sealed_references_path: Path, config: Any, output_dir: Path) -> dict[str, Any]:
    """Validate both outputs completely before opening local-only references and scoring."""
    _validate_sidecar(account_a_bin); _validate_sidecar(account_b_bin)
    temp = Path(tempfile.mkdtemp(prefix="e45_final_"))
    try:
        a_root, b_root = temp / "account_a", temp / "account_b"
        safe_extract_archive(account_a_bin, a_root); safe_extract_archive(account_b_bin, b_root)
        _validate_manifest(a_root, "Account A"); _validate_manifest(b_root, "Account B")
        contexts_a, contexts_b = _jsonl(a_root / "holdout-contexts.jsonl"), _jsonl(b_root / "holdout-contexts.jsonl")
        if (a_root / "holdout-contexts.jsonl").read_bytes() != (b_root / "holdout-contexts.jsonl").read_bytes():
            raise FinalizeError("Account A/B prepared contexts are not byte-identical")
        qids = _validate_contexts(contexts_a)
        raw_a, clean_a, runtime_a, _ = _validate_arm(root=a_root, label="Account A", contexts=contexts_a, qids=qids, expected_control=False)
        raw_b, clean_b, runtime_b, _ = _validate_arm(root=b_root, label="Account B", contexts=contexts_a, qids=qids, expected_control=True)
        # Hai archive phải cùng khớp contract local đã chốt mà merger nhận vào.
        # Chỉ khớp với nhau thì chưa đủ.
        local_project_root = Path(__file__).resolve().parents[2]
        expected_config_sha = getattr(config, "sha", getattr(config, "sha256", ""))
        expected_code_sha = compute_source_identity(local_project_root)
        expected_model = config.section("model_inventory")
        expected_runtime = config.section("runtime")
        expected_decoding = compute_json_sha256(config.section("inference"))
        expected_cleanup = compute_json_sha256(config.section("unified_clean"))
        for runtime, label in ((runtime_a, "Account A"), (runtime_b, "Account B")):
            if (
                runtime["config_sha256"] != expected_config_sha
                or runtime["code_sha256"] != expected_code_sha
                or runtime["base_model"] != expected_model["base_model"]
                or runtime["base_revision"] != expected_model["base_revision"]
                or runtime["decoding_sha256"] != expected_decoding
                or runtime["cleanup_sha256"] != expected_cleanup
                or any(runtime["runtime_versions"].get(name) != version for name, version in expected_runtime.items())
            ):
                raise FinalizeError(f"{label} runtime identity does not match the local frozen E45 contract")
        same_fields = ("experiment_id", "config_sha256", "base_model", "base_revision", "tokenizer_sha256", "decoding_sha256", "cleanup_sha256", "context_artifact_sha256")
        if any(runtime_a[field] != runtime_b[field] for field in same_fields):
            raise FinalizeError("Account A/B runtime identities differ outside the adapter")
        complete = json.loads((a_root / "complete.json").read_text(encoding="utf-8"))
        adapter = a_root / "adapter" / "adapter_model.safetensors"
        adapter_config = a_root / "adapter" / "adapter_config.json"
        complete_required = {
            "experiment_id", "config_sha256", "global_step", "expected_total_steps", "train_loss",
            "adapter_model_sha256", "adapter_config_sha256", "code_manifest_sha256",
            "training_records_jsonl_sha256", "aggregate_records_sha256", "tokenizer_sha256",
            "tokenizer_aggregate_sha256", "base_model", "base_revision", "lora_rank", "lora_alpha",
            "lora_dropout", "target_modules", "runtime_versions", "completed_at_utc",
        }
        if set(complete) != complete_required:
            raise FinalizeError("Candidate complete.json has an unexpected or incomplete schema")
        if (
            not adapter.is_file()
            or not adapter_config.is_file()
            or complete.get("adapter_model_sha256") != file_sha256(adapter)
            or complete.get("adapter_config_sha256") != file_sha256(adapter_config)
            or runtime_a["adapter_sha256"] != file_sha256(adapter)
            or complete["experiment_id"] != runtime_a["experiment_id"]
            or complete["config_sha256"] != runtime_a["config_sha256"]
            or complete["code_manifest_sha256"] != runtime_a["code_sha256"]
            or complete["base_model"] != runtime_a["base_model"]
            or complete["base_revision"] != runtime_a["base_revision"]
            or complete["tokenizer_sha256"] != runtime_a["tokenizer_sha256"]
            or complete["runtime_versions"] != runtime_a["runtime_versions"]
            or complete["global_step"] != complete["expected_total_steps"]
            or complete["expected_total_steps"] != 705
        ):
            raise FinalizeError("Candidate complete.json/runtime adapter identity mismatch")
        if not sealed_references_path.is_file() or file_sha256(sealed_references_path) != PINNED_SEALED_REFERENCES_SHA256:
            raise FinalizeError("Sealed local reference authority is missing or changed")
        scorer_identity = _verify_official_scorer_runtime()
        refs = json.loads(sealed_references_path.read_text(encoding="utf-8"))
        if set(refs) != set(qids):
            raise FinalizeError("Sealed references do not cover the frozen holdout")

        meteor_delta, rouge_delta, cand_meteor, ctrl_meteor, cand_rouge, ctrl_rouge = [], [], [], [], [], []
        for index, qid in enumerate(qids):
            reference = refs[qid]["answer"]
            cm, bm = compute_official_meteor(reference, clean_a[index]["clean_answer"]), compute_official_meteor(reference, clean_b[index]["clean_answer"])
            cr, br = compute_official_rouge_l(reference, clean_a[index]["clean_answer"]), compute_official_rouge_l(reference, clean_b[index]["clean_answer"])
            cand_meteor.append(cm); ctrl_meteor.append(bm); meteor_delta.append(cm - bm)
            cand_rouge.append(cr); ctrl_rouge.append(br); rouge_delta.append(cr - br)
        ci = _bootstrap(meteor_delta)
        cand_length = sum(row["finish_reason"] == "length" for row in raw_a) / 200.0
        ctrl_length = sum(row["finish_reason"] == "length" for row in raw_b) / 200.0
        mean_meteor, mean_rouge = float(np.mean(meteor_delta)), float(np.mean(rouge_delta))
        passed = mean_meteor >= 0.010 and ci[0] > 0.0 and mean_rouge >= 0.0 and cand_length < ctrl_length
        prompt_lengths = [row["input_tokens"] for row in contexts_a]
        # Lượt hai sinh lại từ đầu khi lượt một chạm trần, không nối tiếp.
        # Chỉ báo độ dài answer cuối; không cộng độ dài hai lượt.
        output_tokens_a = [row["second_pass_tokens"] or row["first_pass_tokens"] for row in raw_a]
        output_tokens_b = [row["second_pass_tokens"] or row["first_pass_tokens"] for row in raw_b]
        summary = {
            "verdict": "E45_PASSES_HELDOUT_GATE_PENDING_LEAD_REVIEW" if passed else "REJECT_E45",
            # Chỉ xuất giá trị này sau khi mọi phép kiểm tra phía trên đạt.
            # Giá trị được suy từ kết quả kiểm tra, không gán cứng.
            "zero_violations": bool(qids and len(raw_a) == len(clean_a) == len(raw_b) == len(clean_b) == 200),
            "metrics": {
                "candidate_meteor_macro": float(np.mean(cand_meteor)), "control_meteor_macro": float(np.mean(ctrl_meteor)),
                "mean_paired_meteor_delta": mean_meteor, "median_paired_meteor_delta": float(np.median(meteor_delta)),
                "worst_paired_meteor_delta": float(np.min(meteor_delta)), "best_paired_meteor_delta": float(np.max(meteor_delta)),
                "bootstrap_95_ci": list(ci), "candidate_rougel_macro": float(np.mean(cand_rouge)),
                "control_rougel_macro": float(np.mean(ctrl_rouge)), "mean_paired_rougel_delta": mean_rouge,
                "improved_count": sum(delta > 1e-6 for delta in meteor_delta), "tied_count": sum(abs(delta) <= 1e-6 for delta in meteor_delta), "worsened_count": sum(delta < -1e-6 for delta in meteor_delta),
            },
            "length_finishes": {
                "candidate_initial_1024_length_finishes": sum(row["initial_finish_reason"] == "length" for row in raw_a),
                "control_initial_1024_length_finishes": sum(row["initial_finish_reason"] == "length" for row in raw_b),
                "candidate_final_1536_length_finishes": int(cand_length * 200),
                "control_final_1536_length_finishes": int(ctrl_length * 200),
            },
            "distributions": {
                "prompt_tokens": {"mean": float(np.mean(prompt_lengths)), "median": float(np.median(prompt_lengths)), "max": int(max(prompt_lengths))},
                "candidate_output_tokens": {"mean": float(np.mean(output_tokens_a)), "median": float(np.median(output_tokens_a)), "max": int(max(output_tokens_a))},
                "control_output_tokens": {"mean": float(np.mean(output_tokens_b)), "median": float(np.median(output_tokens_b)), "max": int(max(output_tokens_b))},
                "candidate_characters": {"mean": float(np.mean([len(row["clean_answer"]) for row in clean_a])), "max": max(len(row["clean_answer"]) for row in clean_a)},
                "control_characters": {"mean": float(np.mean([len(row["clean_answer"]) for row in clean_b])), "max": max(len(row["clean_answer"]) for row in clean_b)},
            },
            "scorer_identity": scorer_identity,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "e45-heldout-decision.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return summary
    finally:
        shutil.rmtree(temp, ignore_errors=True)


def merge_and_score_e45(**kwargs: Any) -> dict[str, Any]:
    """Backward-compatible name for the only supported strict merger."""
    return merge_and_evaluate_e45(**kwargs)
