"""Fail-closed four-shard private inference helpers for an admitted E45 adapter.

This module deliberately reuses E45's frozen retrieval, prompt rendering, and
P01 generation implementation.  It only adds global-index sharding, admission
binding, and local assembly; it never reads private reference answers.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

from e45_private_checkpoint import (
    PrivateCheckpointError,
    find_private_resume_checkpoint,
    restore_private_checkpoint,
)
from e45_private_resumable_generation import (
    PrivateGenerationResumeError,
    execute_resumable_private_p01_generation,
)
from uit_dsc_fixed_rag.corpus import file_sha256
from uit_dsc_fixed_rag.e45_checkpoint import safe_extract_archive


PRIVATE_SHARD_EXPERIMENT = "E45-private-four-shards-r1"
ADMISSION_SCHEMA = "E45_PRIVATE_ADMISSION_R1"
DIRECT_RELEASE_SCHEMA = "E45_PRIVATE_DIRECT_RELEASE_R1"
DIRECT_RELEASE_AUTHORIZATION = "user-authorized-direct-private-after-complete-candidate"
PRIVATE_SHA256 = "d84bce10a1aa1c939552b2d86d0f8cb7b66d2d6ed20266e3c2b80f3498b3d427"
PRIVATE_SAMPLE_COUNT = 1918
PRIVATE_SAMPLE_IDS_SHA256 = "2840be37e59873e735610177cf7caffe78e14e70b7cc48d8ac1e57545d15bb8e"
PRIVATE_EXECUTION_STATUS_SCHEMA = "E45_PRIVATE_SHARD_EXECUTION_STATUS_R4"


class PrivateShardError(RuntimeError):
    """Raised when private execution or assembly loses an immutable binding."""


def compute_json_sha256(data: Any) -> str:
    """Hash the canonical JSON representation used by the frozen E45 records."""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _ids_sha256(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrivateShardError(f"Invalid JSON file: {path.name}") from exc
    if not isinstance(value, dict):
        raise PrivateShardError(f"Expected a JSON object: {path.name}")
    return value


def load_private_questions(path: Path) -> tuple[dict[str, str], list[str], dict[str, Any]]:
    """Load question-only private data and reject every non-null answer."""
    path = Path(path)
    if not path.is_file() or file_sha256(path) != PRIVATE_SHA256:
        raise PrivateShardError("Private question artifact does not match its pinned SHA-256")
    payload = _read_json(path)
    questions: dict[str, str] = {}
    for raw_qid, record in payload.items():
        qid = str(raw_qid)
        if not isinstance(record, dict) or set(record) not in ({"question"}, {"question", "answer"}):
            raise PrivateShardError(f"Invalid private question record schema: {qid}")
        question = record.get("question")
        if not isinstance(question, str) or not question.strip():
            raise PrivateShardError(f"Empty private question: {qid}")
        if "answer" in record and record["answer"] is not None:
            raise PrivateShardError(f"Private reference answer is present: {qid}")
        if qid in questions:
            raise PrivateShardError(f"Duplicate private question ID: {qid}")
        questions[qid] = question
    ids = sorted(questions)
    if len(ids) != PRIVATE_SAMPLE_COUNT:
        raise PrivateShardError(f"Private sample count {len(ids)} != {PRIVATE_SAMPLE_COUNT}")
    if _ids_sha256(ids) != PRIVATE_SAMPLE_IDS_SHA256:
        raise PrivateShardError("Private question IDs do not match the frozen lexicographic order")
    return questions, ids, {
        "private_sha256": file_sha256(path),
        "sample_ids_sha256": _ids_sha256(ids),
        "sample_size": len(ids),
        "private_reference_answers_read": False,
    }


def load_admission(path: Path) -> dict[str, Any]:
    """Validate either a heldout-pass or explicitly authorized direct-release token."""
    value = _read_json(path)
    common = {
        "schema", "experiment_id", "heldout_verdict", "heldout_decision_sha256",
        "config_sha256", "candidate_archive_sha256", "candidate_adapter_sha256",
        "candidate_adapter_config_sha256", "candidate_complete_sha256", "private_sha256",
        "private_sample_ids_sha256", "private_sample_size", "issued_at_utc", "admission_sha256",
    }
    direct = {
        "schema", "experiment_id", "authorization", "config_sha256", "candidate_archive_sha256",
        "candidate_adapter_sha256", "candidate_adapter_config_sha256", "candidate_complete_sha256",
        "private_sha256", "private_sample_ids_sha256", "private_sample_size", "issued_at_utc",
        "admission_sha256",
    }
    schema = value.get("schema")
    if (schema == ADMISSION_SCHEMA and set(value) != common) or (schema == DIRECT_RELEASE_SCHEMA and set(value) != direct):
        raise PrivateShardError("Private admission has an unexpected schema")
    if schema not in {ADMISSION_SCHEMA, DIRECT_RELEASE_SCHEMA}:
        raise PrivateShardError("Private admission schema is not recognized")
    body = {key: item for key, item in value.items() if key != "admission_sha256"}
    if value["admission_sha256"] != compute_json_sha256(body):
        raise PrivateShardError("Private admission hash is invalid")
    if value["experiment_id"] != "E45-inference-aligned-parent-lora-v1":
        raise PrivateShardError("Private admission experiment identity changed")
    if schema == ADMISSION_SCHEMA and value["heldout_verdict"] != "E45_PASSES_HELDOUT_GATE_PENDING_LEAD_REVIEW":
        raise PrivateShardError("Private inference is blocked because E45 did not pass the heldout gate")
    if schema == DIRECT_RELEASE_SCHEMA and value["authorization"] != DIRECT_RELEASE_AUTHORIZATION:
        raise PrivateShardError("Private direct-release authorization is invalid")
    if value["private_sha256"] != PRIVATE_SHA256 or value["private_sample_size"] != PRIVATE_SAMPLE_COUNT:
        raise PrivateShardError("Private admission does not bind the official private artifact")
    return value


def verify_admission_inputs(*, admission: dict[str, Any], config_path: Path, private_path: Path) -> dict[str, Any]:
    """Bind the admission token to this exact E45 config and private question file."""
    _, ids, private_identity = load_private_questions(private_path)
    if (
        admission["config_sha256"] != file_sha256(config_path)
        or admission["private_sha256"] != private_identity["private_sha256"]
        or admission["private_sample_ids_sha256"] != _ids_sha256(ids)
    ):
        raise PrivateShardError("Admission/config/private identity mismatch")
    return private_identity


def _validate_sidecar(archive: Path, expected_sha256: str) -> None:
    """Verify archive bytes and one matching nearby SHA-256 sidecar.

    Browser downloads can rename ZIP-bytes bin files to zip. The sidecar is
    resolved by declared digest within the supplied artifact directory, while
    ambiguity still fails closed.
    """
    archive = Path(archive)
    actual = file_sha256(archive)
    if actual != expected_sha256:
        raise PrivateShardError(f"Archive bytes do not match expected SHA-256: {archive.name}")
    candidates = [Path(str(archive) + ".sha256")]
    candidates.extend(path for path in archive.parent.glob("*.sha256") if path not in candidates)
    matches: list[Path] = []
    for sidecar in candidates:
        if not sidecar.is_file():
            continue
        try:
            fields = sidecar.read_text(encoding="utf-8").strip().split()
        except OSError as exc:
            raise PrivateShardError(f"Cannot read archive SHA-256 sidecar: {sidecar.name}") from exc
        if fields and fields[0].lower() == expected_sha256:
            matches.append(sidecar)
    if len(matches) != 1:
        raise PrivateShardError(
            f"Expected exactly one matching SHA-256 sidecar for {archive.name}; found {len(matches)}"
        )


def _validate_archive_manifest(root: Path) -> None:
    manifest = _read_json(root / "manifest.json")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != set(manifest) | {"manifest.json"}:
        raise PrivateShardError("Candidate archive members do not match its manifest")
    for relative, metadata in manifest.items():
        path = root / relative
        if not path.is_file() or path.stat().st_size != metadata.get("bytes") or file_sha256(path) != metadata.get("sha256"):
            raise PrivateShardError(f"Candidate archive manifest mismatch: {relative}")


def materialize_candidate_adapter(*, candidate_archive: Path, destination: Path, admission: dict[str, Any]) -> Path:
    """Safely extract and pin the admitted E45 adapter in a two-file PEFT root."""
    candidate_archive = Path(candidate_archive)
    _validate_sidecar(candidate_archive, admission["candidate_archive_sha256"])
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise PrivateShardError("Candidate extraction destination must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="e45_private_candidate_") as temporary:
        extracted = Path(temporary) / "candidate"
        safe_extract_archive(candidate_archive, extracted)
        _validate_archive_manifest(extracted)
        complete_path = extracted / "complete.json"
        model_path = extracted / "adapter" / "adapter_model.safetensors"
        config_path = extracted / "adapter" / "adapter_config.json"
        if not all(path.is_file() for path in (complete_path, model_path, config_path)):
            raise PrivateShardError("Candidate archive lacks complete E45 adapter files")
        if (
            file_sha256(complete_path) != admission["candidate_complete_sha256"]
            or file_sha256(model_path) != admission["candidate_adapter_sha256"]
            or file_sha256(config_path) != admission["candidate_adapter_config_sha256"]
        ):
            raise PrivateShardError("Candidate archive differs from the admitted adapter")
        complete = _read_json(complete_path)
        if (
            complete.get("experiment_id") != "E45-inference-aligned-parent-lora-v1"
            or complete.get("global_step") != 705
            or complete.get("expected_total_steps") != 705
            or complete.get("adapter_model_sha256") != admission["candidate_adapter_sha256"]
            or complete.get("adapter_config_sha256") != admission["candidate_adapter_config_sha256"]
        ):
            raise PrivateShardError("Candidate complete.json is not a completed admitted E45 run")
        shutil.copy2(model_path, destination / "adapter_model.safetensors")
        shutil.copy2(config_path, destination / "adapter_config.json")
    observed = {path.name for path in destination.iterdir() if path.is_file()}
    if observed != {"adapter_model.safetensors", "adapter_config.json"}:
        raise PrivateShardError("Controlled candidate adapter directory has unexpected files")
    return destination


def shard_questions(*, private_path: Path, shard_index: int, shard_count: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Partition sorted private IDs by immutable global index modulo shard count."""
    if shard_count != 4 or shard_index not in range(shard_count):
        raise PrivateShardError("E45 private execution requires exactly shard indexes 0..3")
    questions, ids, identity = load_private_questions(private_path)
    assigned = [(index, qid) for index, qid in enumerate(ids) if index % shard_count == shard_index]
    rows = [{"question_id": qid, "question": questions[qid]} for _, qid in assigned]
    shard_identity = {
        **identity,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "global_indices": [index for index, _ in assigned],
        "global_question_ids": [qid for _, qid in assigned],
    }
    shard_identity["shard_identity_sha256"] = compute_json_sha256(shard_identity)
    return rows, shard_identity


def prepare_private_shard_contexts(
    *, private_path: Path, shard_index: int, shard_count: int, e00_dir: Path,
    dense_dir: Path, tokenizer_path: Path, output_dir: Path, config_path: Path,
) -> tuple[Path, dict[str, Any]]:
    """Prepare answer-blind E45 P00/P01 contexts for one global private shard."""
    from uit_dsc_fixed_rag.e45_holdout_contexts import prepare_holdout_contexts
    from uit_dsc_fixed_rag.e45_parent_training import load_config

    config = load_config(config_path)
    questions, shard_identity = shard_questions(
        private_path=private_path, shard_index=shard_index, shard_count=shard_count
    )
    contexts_path, _ = prepare_holdout_contexts(
        questions=questions,
        e00_dir=e00_dir,
        dense_dir=dense_dir,
        tokenizer_path=tokenizer_path,
        output_dir=output_dir,
        config=config,
        device="cuda:0",
    )
    # Iterate physical JSONL records. ``str.splitlines()`` also splits valid
    # JSON strings at Unicode separators such as U+2028, which occurs in the
    # official corpus and can make an otherwise valid record look truncated.
    with contexts_path.open("r", encoding="utf-8", newline="") as stream:
        contexts = [json.loads(line) for line in stream if line.strip()]
    if [row.get("question_id") for row in contexts] != shard_identity["global_question_ids"]:
        raise PrivateShardError("Private context QIDs differ from the assigned global shard")
    manifest = {
        "schema": "E45_PRIVATE_CONTEXT_SHARD_R1",
        "experiment_id": config.experiment_id,
        "config_sha256": config.sha256,
        "contexts_sha256": file_sha256(contexts_path),
        "context_count": len(contexts),
        "answers_used": False,
        **shard_identity,
    }
    _atomic_json(output_dir / "private-context-manifest.json", manifest)
    return contexts_path, manifest


def run_private_shard(
    *, admission_path: Path, candidate_archive: Path, private_path: Path,
    shard_index: int, shard_count: int, e00_dir: Path, dense_dir: Path,
    tokenizer_path: Path, output_dir: Path, config_path: Path,
    checkpoint_archive: Path, resume_checkpoint: Path | None = None,
    wall_clock_seconds: int | None = None,
) -> dict[str, Any]:
    """Run or resume one admitted private shard at an atomic P01 record boundary.

    A resume archive restores only generation state.  Contexts are rebuilt via
    the canonical answer-blind E45 path and their hash must match the stored
    checkpoint before any saved answer is used.
    """
    from uit_dsc_fixed_rag.e45_paired_generation import E45GeneratorWorker
    from uit_dsc_fixed_rag.e45_parent_training import load_config
    from uit_dsc_fixed_rag.e45_input_resolver import compute_source_identity

    admission = load_admission(admission_path)
    private_identity = verify_admission_inputs(
        admission=admission, config_path=config_path, private_path=private_path
    )
    config = load_config(config_path)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise PrivateShardError("Private shard output directory must be fresh")
    output_dir.mkdir(parents=True)
    checkpoint_archive = Path(checkpoint_archive)
    if checkpoint_archive.suffix != ".bin":
        raise PrivateShardError("Private shard checkpoint archive must use a .bin filename")
    if wall_clock_seconds is not None and wall_clock_seconds <= 0:
        raise PrivateShardError("Private shard wall-clock safety limit must be positive")
    adapter = materialize_candidate_adapter(
        candidate_archive=candidate_archive, destination=output_dir / "candidate-adapter", admission=admission
    )
    contexts_path, context_manifest = prepare_private_shard_contexts(
        private_path=private_path, shard_index=shard_index, shard_count=shard_count,
        e00_dir=e00_dir, dense_dir=dense_dir, tokenizer_path=tokenizer_path,
        output_dir=output_dir / "contexts", config_path=config_path,
    )
    # Iterate physical JSONL records. ``str.splitlines()`` also splits valid
    # JSON strings at Unicode separators such as U+2028, which occurs in the
    # official corpus and can make an otherwise valid record look truncated.
    with contexts_path.open("r", encoding="utf-8", newline="") as stream:
        contexts = [json.loads(line) for line in stream if line.strip()]
    worker = E45GeneratorWorker(
        base_model_path=config.section("model_inventory")["base_model"],
        tokenizer_path=tokenizer_path,
        adapter_path=adapter,
        config=config,
    )
    context_manifest_path = output_dir / "contexts" / "private-context-manifest.json"
    checkpoint_identity = {
        "schema": "E45_PRIVATE_CHECKPOINT_IDENTITY_R4",
        "experiment_id": config.experiment_id,
        "admission_sha256": admission["admission_sha256"],
        "config_sha256": config.sha256,
        "system_code_sha256": compute_source_identity(Path(config_path).resolve().parent.parent),
        "private_scheduler_sha256": file_sha256(Path(__file__)),
        "candidate_archive_sha256": admission["candidate_archive_sha256"],
        "candidate_adapter_sha256": admission["candidate_adapter_sha256"],
        "candidate_adapter_config_sha256": admission["candidate_adapter_config_sha256"],
        "private_sha256": private_identity["private_sha256"],
        "private_sample_ids_sha256": private_identity["sample_ids_sha256"],
        "private_sample_size": private_identity["sample_size"],
        "shard_index": shard_index,
        "shard_count": shard_count,
        "shard_identity_sha256": context_manifest["shard_identity_sha256"],
        "global_question_ids_sha256": _ids_sha256(context_manifest["global_question_ids"]),
        "contexts_sha256": file_sha256(contexts_path),
        "context_manifest_sha256": file_sha256(context_manifest_path),
        "generation_identity": worker.identity(),
    }
    generation_dir = output_dir / "generation"
    if resume_checkpoint is not None:
        try:
            restore_private_checkpoint(
                archive_path=Path(resume_checkpoint),
                destination_state_dir=generation_dir / "worker-states",
                expected_identity=checkpoint_identity,
            )
        except PrivateCheckpointError as exc:
            raise PrivateShardError("Private resume checkpoint is incompatible or altered") from exc
    deadline_epoch = time.time() + wall_clock_seconds if wall_clock_seconds is not None else None
    try:
        outcome = execute_resumable_private_p01_generation(
            worker=worker,
            contexts=contexts,
            output_dir=generation_dir,
            arm_name="private-candidate",
            checkpoint_archive=checkpoint_archive,
            checkpoint_identity=checkpoint_identity,
            deadline_epoch=deadline_epoch,
        )
    except PrivateGenerationResumeError as exc:
        raise PrivateShardError("Private P01 generation cannot safely continue") from exc
    execution_status = {
        "schema": PRIVATE_EXECUTION_STATUS_SCHEMA,
        "experiment_id": config.experiment_id,
        "admission_sha256": admission["admission_sha256"],
        "checkpoint_archive": outcome.checkpoint["checkpoint_archive"],
        "checkpoint_archive_sha256": outcome.checkpoint["checkpoint_archive_sha256"],
        "checkpoint_identity_sha256": outcome.checkpoint["checkpoint_identity_sha256"],
        "completed_first_pass": outcome.completed_first_pass,
        "total": outcome.total,
        "status": "COMPLETE" if outcome.completed else "CHECKPOINTED",
        "reason": outcome.reason,
        "private_reference_answers_read": False,
    }
    if not outcome.completed:
        execution_status["execution_status_sha256"] = compute_json_sha256(execution_status)
        _atomic_json(output_dir / "execution-status.json", execution_status)
        return execution_status
    if outcome.raw_path is None or outcome.clean_path is None or outcome.report is None:
        raise PrivateShardError("Completed private generation lacks final records")
    raw_path, clean_path, report = outcome.raw_path, outcome.clean_path, outcome.report
    report_path = output_dir / "generation" / "generation-report.json"
    shard_report = {
        "schema": "E45_PRIVATE_SHARD_REPORT_R1",
        "experiment_id": config.experiment_id,
        "admission_sha256": admission["admission_sha256"],
        "config_sha256": config.sha256,
        "candidate_archive_sha256": admission["candidate_archive_sha256"],
        "candidate_adapter_sha256": admission["candidate_adapter_sha256"],
        "private_identity": private_identity,
        "context_manifest_sha256": file_sha256(context_manifest_path),
        "contexts_sha256": file_sha256(contexts_path),
        "raw_records_sha256": file_sha256(raw_path),
        "clean_records_sha256": file_sha256(clean_path),
        "generation_report_sha256": file_sha256(report_path),
        "generation_identity": report["identity"],
        "shard_identity": {key: context_manifest[key] for key in (
            "shard_index", "shard_count", "global_indices", "global_question_ids", "shard_identity_sha256"
        )},
        "private_reference_answers_read": False,
    }
    shard_report["shard_report_sha256"] = compute_json_sha256(shard_report)
    _atomic_json(output_dir / "shard-report.json", shard_report)
    execution_status.update({
        "shard_report_sha256": shard_report["shard_report_sha256"],
        "raw_records_sha256": shard_report["raw_records_sha256"],
        "clean_records_sha256": shard_report["clean_records_sha256"],
    })
    execution_status["execution_status_sha256"] = compute_json_sha256(execution_status)
    _atomic_json(output_dir / "execution-status.json", execution_status)
    return execution_status


def _file_manifest(root: Path, excluded: set[str]) -> dict[str, dict[str, Any]]:
    return {
        path.relative_to(root).as_posix(): {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    }


def package_shard(*, output_dir: Path, archive_path: Path) -> tuple[Path, str]:
    """Package the complete identity-bound shard as a safe ZIP-bytes .bin."""
    output_dir, archive_path = Path(output_dir), Path(archive_path)
    required = (
        output_dir / "contexts" / "holdout-contexts.jsonl",
        output_dir / "contexts" / "private-context-manifest.json",
        output_dir / "generation" / "raw-records.jsonl",
        output_dir / "generation" / "clean-records.jsonl",
        output_dir / "generation" / "generation-report.json",
        output_dir / "shard-report.json",
    )
    if not all(path.is_file() for path in required):
        raise PrivateShardError("Cannot package incomplete private shard")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    # A fresh external staging directory makes the final packaging cell safe
    # to rerun after a browser reconnect or notebook-cell retry.
    with tempfile.TemporaryDirectory(prefix="e45_private_bundle_", dir=archive_path.parent) as temporary_root:
        stage = Path(temporary_root) / "bundle"
        stage.mkdir()
        for source, target in (
            (output_dir / "contexts", stage / "contexts"),
            (output_dir / "generation", stage / "generation"),
        ):
            shutil.copytree(source, target)
        shutil.copy2(output_dir / "shard-report.json", stage / "shard-report.json")
        manifest = _file_manifest(stage, set())
        _atomic_json(stage / "FILE_MANIFEST.json", manifest)
        manifest["FILE_MANIFEST.json"] = {
            "bytes": (stage / "FILE_MANIFEST.json").stat().st_size,
            "sha256": file_sha256(stage / "FILE_MANIFEST.json"),
        }
        temporary = archive_path.with_suffix(".tmp")
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for relative in sorted(manifest):
                archive.write(stage / relative, relative)
        temporary.replace(archive_path)
    digest = file_sha256(archive_path)
    Path(str(archive_path) + ".sha256").write_text(f"{digest}  {archive_path.name}\n", encoding="ascii")
    return archive_path, digest


def _validate_bundle(root: Path) -> dict[str, Any]:
    manifest = _read_json(root / "FILE_MANIFEST.json")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != set(manifest) | {"FILE_MANIFEST.json"}:
        raise PrivateShardError("Shard bundle members differ from FILE_MANIFEST")
    for relative, details in manifest.items():
        path = root / relative
        if not path.is_file() or path.stat().st_size != details.get("bytes") or file_sha256(path) != details.get("sha256"):
            raise PrivateShardError(f"Shard bundle manifest mismatch: {relative}")
    report = _read_json(root / "shard-report.json")
    body = {key: value for key, value in report.items() if key != "shard_report_sha256"}
    if report.get("shard_report_sha256") != compute_json_sha256(body):
        raise PrivateShardError("Shard report hash mismatch")
    return report


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL artifact strictly, preserving its persisted record order."""
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise PrivateShardError(f"Invalid JSONL artifact: {path.name}") from exc


def _validate_shard_execution(
    *, root: Path, report: dict[str, Any], admission: dict[str, Any], expected_ids: list[str]
) -> list[dict[str, Any]]:
    """Verify every context/raw/clean binding before an answer reaches submission."""
    required_report = {
        "schema", "experiment_id", "admission_sha256", "config_sha256", "candidate_archive_sha256",
        "candidate_adapter_sha256", "private_identity", "context_manifest_sha256", "contexts_sha256",
        "raw_records_sha256", "clean_records_sha256", "generation_report_sha256", "generation_identity",
        "shard_identity", "private_reference_answers_read", "shard_report_sha256",
    }
    if set(report) != required_report or report["schema"] != "E45_PRIVATE_SHARD_REPORT_R1":
        raise PrivateShardError("Shard report schema changed")
    if (
        report["experiment_id"] != admission["experiment_id"]
        or report["config_sha256"] != admission["config_sha256"]
        or report["candidate_archive_sha256"] != admission["candidate_archive_sha256"]
        or report["candidate_adapter_sha256"] != admission["candidate_adapter_sha256"]
        or report["private_reference_answers_read"] is not False
    ):
        raise PrivateShardError("Shard report has an incompatible admission identity")
    contexts_path = root / "contexts" / "holdout-contexts.jsonl"
    context_manifest_path = root / "contexts" / "private-context-manifest.json"
    raw_path = root / "generation" / "raw-records.jsonl"
    clean_path = root / "generation" / "clean-records.jsonl"
    generation_report_path = root / "generation" / "generation-report.json"
    for path, expected_hash, name in (
        (contexts_path, report["contexts_sha256"], "contexts"),
        (context_manifest_path, report["context_manifest_sha256"], "context manifest"),
        (raw_path, report["raw_records_sha256"], "raw records"),
        (clean_path, report["clean_records_sha256"], "clean records"),
        (generation_report_path, report["generation_report_sha256"], "generation report"),
    ):
        if not path.is_file() or file_sha256(path) != expected_hash:
            raise PrivateShardError(f"Shard {name} hash mismatch")
    context_manifest = _read_json(context_manifest_path)
    if (
        context_manifest.get("contexts_sha256") != report["contexts_sha256"]
        or context_manifest.get("context_count") != len(expected_ids)
        or context_manifest.get("answers_used") is not False
        or context_manifest.get("global_question_ids") != expected_ids
    ):
        raise PrivateShardError("Private context manifest is not bound to this shard")
    generation_report = _read_json(generation_report_path)
    states_dir = root / "generation" / "worker-states"
    expected_state_files = generation_report.get("worker_state_files")
    if not isinstance(expected_state_files, list) or not expected_state_files:
        raise PrivateShardError("Generation report does not bind worker state files")
    actual_state_files = sorted(path.name for path in states_dir.glob("*.json")) if states_dir.is_dir() else []
    if sorted(expected_state_files) != actual_state_files:
        raise PrivateShardError("Generation worker-state file set differs from report")
    state_raw_hashes: set[str] = set()
    for name in actual_state_files:
        state = _read_json(states_dir / name)
        if state.get("status") not in {"complete", "oom"} or state.get("identity") != generation_report.get("identity"):
            raise PrivateShardError(f"Worker state is incomplete or has incompatible identity: {name}")
        completed = state.get("completed")
        raw_hashes = state.get("raw_record_sha256s", [])
        if not isinstance(completed, list) or not isinstance(raw_hashes, list) or len(completed) != len(raw_hashes):
            raise PrivateShardError(f"Worker state has invalid completion binding: {name}")
        if len(completed) != len(set(completed)) or len(raw_hashes) != len(set(raw_hashes)):
            raise PrivateShardError(f"Worker state has duplicate completion binding: {name}")
        state_raw_hashes.update(raw_hashes)
    if (
        generation_report.get("arm_name") != "private-candidate"
        or generation_report.get("sample_size") != len(expected_ids)
        or generation_report.get("raw_records_sha256") != report["raw_records_sha256"]
        or generation_report.get("clean_records_sha256") != report["clean_records_sha256"]
        or generation_report.get("identity") != report["generation_identity"]
        or generation_report.get("identity", {}).get("adapter_sha256") != admission["candidate_adapter_sha256"]
        or generation_report.get("identity", {}).get("config_sha256") != admission["config_sha256"]
    ):
        raise PrivateShardError("Generation report is not bound to the admitted candidate")
    contexts, raw_rows, clean_rows = _read_jsonl(contexts_path), _read_jsonl(raw_path), _read_jsonl(clean_path)
    raw_hashes = {row.get("record_sha256") for row in raw_rows}
    if state_raw_hashes != raw_hashes:
        raise PrivateShardError("Worker states do not bind exactly the raw generation records")
    if not (len(contexts) == len(raw_rows) == len(clean_rows) == len(expected_ids)):
        raise PrivateShardError("Shard contexts/raw/clean counts do not match")
    for index, (qid, context, raw, clean) in enumerate(zip(expected_ids, contexts, raw_rows, clean_rows)):
        context_body = {key: value for key, value in context.items() if key != "record_sha256"}
        raw_body = {key: value for key, value in raw.items() if key != "record_sha256"}
        clean_body = {key: value for key, value in clean.items() if key != "record_sha256"}
        if (
            context.get("sample_index") != index or raw.get("sample_index") != index or clean.get("sample_index") != index
            or context.get("question_id") != qid or raw.get("question_id") != qid or clean.get("question_id") != qid
            or context.get("answers_used") is not False
            or context.get("record_sha256") != compute_json_sha256(context_body)
            or raw.get("record_sha256") != compute_json_sha256(raw_body)
            or clean.get("record_sha256") != compute_json_sha256(clean_body)
            or raw.get("context_record_sha256") != context.get("record_sha256")
            or raw.get("question_sha256") != context.get("question_sha256")
            or raw.get("evidence_sha256") != context.get("evidence_sha256")
            or raw.get("prompt_sha256") != context.get("prompt_sha256")
            or raw.get("prompt_input_ids_sha256") != context.get("prompt_input_ids_sha256")
            or raw.get("adapter_sha256") != admission["candidate_adapter_sha256"]
            or clean.get("raw_record_sha256") != raw.get("record_sha256")
            or clean.get("cleanup_sha256") != generation_report["identity"].get("cleanup_sha256")
        ):
            raise PrivateShardError(f"Shard record binding mismatch at local index {index}")
    return clean_rows


def merge_private_shards(*, admission_path: Path, private_path: Path, shard_archives: list[Path], output_dir: Path) -> Path:
    """Verify four complete outputs and produce the competition-only submission.zip."""
    admission = load_admission(admission_path)
    _, ids, private_identity = load_private_questions(private_path)
    if admission["private_sample_ids_sha256"] != private_identity["sample_ids_sha256"]:
        raise PrivateShardError("Admission/private IDs differ during local merge")
    if len(shard_archives) != 4:
        raise PrivateShardError("Exactly four E45 private shard archives are required")
    extracted: list[tuple[Path, dict[str, Any]]] = []
    temporary = Path(tempfile.mkdtemp(prefix="e45_private_merge_"))
    try:
        for ordinal, archive in enumerate(shard_archives):
            archive = Path(archive)
            _validate_sidecar(archive, file_sha256(archive))
            root = temporary / str(ordinal)
            safe_extract_archive(archive, root)
            extracted.append((root, _validate_bundle(root)))
        reports = [report for _, report in extracted]
        if {report.get("admission_sha256") for report in reports} != {admission["admission_sha256"]}:
            raise PrivateShardError("Shard admission identities differ")
        if {report.get("candidate_archive_sha256") for report in reports} != {admission["candidate_archive_sha256"]}:
            raise PrivateShardError("Shard candidate archive identities differ")
        indexed = {report["shard_identity"]["shard_index"]: (root, report) for root, report in extracted}
        if set(indexed) != {0, 1, 2, 3} or len(indexed) != 4:
            raise PrivateShardError("Shard indexes are missing or duplicated")
        answers: dict[str, str] = {}
        for shard_index in range(4):
            root, report = indexed[shard_index]
            identity = report["shard_identity"]
            expected_global = [index for index in range(len(ids)) if index % 4 == shard_index]
            expected_ids = [ids[index] for index in expected_global]
            if identity.get("shard_count") != 4 or identity.get("global_indices") != expected_global or identity.get("global_question_ids") != expected_ids:
                raise PrivateShardError(f"Shard {shard_index} has invalid global-index ownership")
            clean_rows = _validate_shard_execution(
                root=root, report=report, admission=admission, expected_ids=expected_ids
            )
            for local_index, (qid, row) in enumerate(zip(expected_ids, clean_rows)):
                body = {key: value for key, value in row.items() if key != "record_sha256"}
                if row.get("sample_index") != local_index or row.get("question_id") != qid or row.get("record_sha256") != compute_json_sha256(body):
                    raise PrivateShardError(f"Shard {shard_index} clean record identity mismatch at {local_index}")
                answer = row.get("clean_answer")
                if not isinstance(answer, str) or not answer.strip() or qid in answers:
                    raise PrivateShardError(f"Invalid or duplicate answer for private QID {qid}")
                answers[qid] = answer
        if list(sorted(answers)) != ids:
            raise PrivateShardError("Merged private QID coverage is incomplete or reordered")
        payload = json.dumps({qid: {"answer": answers[qid]} for qid in ids}, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        submission = output_dir / "submission.json"
        submission.write_bytes(payload)
        archive_path = output_dir / "submission.zip"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("submission.json", payload)
        with zipfile.ZipFile(archive_path) as archive:
            if archive.namelist() != ["submission.json"] or archive.read("submission.json") != payload:
                raise PrivateShardError("submission.zip contract verification failed")
        _atomic_json(output_dir / "merge-report.json", {
            "schema": "E45_PRIVATE_MERGE_REPORT_R1", "admission_sha256": admission["admission_sha256"],
            "private_sha256": private_identity["private_sha256"], "sample_ids_sha256": private_identity["sample_ids_sha256"],
            "sample_size": len(ids), "submission_sha256": file_sha256(submission),
            "submission_zip_sha256": file_sha256(archive_path), "private_reference_answers_read": False,
        })
        return archive_path
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
