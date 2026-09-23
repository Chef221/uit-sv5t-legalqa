#!/usr/bin/env python3
"""Ghép bản nộp private từ bốn checkpoint đã đủ câu trả lời lượt đầu.

Đây là phương án sát deadline: giữ mọi lượt sinh lại P01 đã hoàn tất và dùng
câu trả lời 1.024 token đã lưu ở nơi lượt sinh lại 1.536 token chưa xong.
Script không đọc đáp án tham chiếu private và không chạy model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e45_private_checkpoint import canonical_sha256  # noqa: E402
from e45_private_shards import (  # noqa: E402
    compute_json_sha256,
    load_admission,
    load_private_questions,
    materialize_candidate_adapter,
    shard_questions,
)
from uit_dsc_fixed_rag.corpus import file_sha256  # noqa: E402
from uit_dsc_fixed_rag.final_private_p01 import unified_clean  # noqa: E402


class EmergencyMergeError(RuntimeError):
    """Checkpoint hoặc bản nộp cuối vi phạm điều kiện bắt buộc."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EmergencyMergeError(message)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _read_checkpoint(path: Path, expected_sha256: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Kiểm tra archive, sidecar đi kèm và từng worker state đã lưu."""
    _require(path.is_file(), f"Checkpoint missing: {path.name}")
    digest = file_sha256(path)
    _require(digest == expected_sha256, f"Checkpoint bytes changed: {path.name}")
    sidecar = Path(str(path) + ".sha256")
    _require(sidecar.is_file(), f"Checkpoint sidecar missing: {sidecar.name}")
    _require(sidecar.read_text(encoding="ascii").split()[0].lower() == digest, "Checkpoint sidecar changed")
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        allowed = {
            "FILE_MANIFEST.json", "PRIVATE_CHECKPOINT_MANIFEST.json",
            "worker-states/sharded-worker-state.json",
            "worker-states/worker-0-state.json", "worker-states/worker-1-state.json",
        }
        _require(set(names) == allowed and len(names) == len(allowed), "Unexpected checkpoint members")
        _require(len({name.casefold() for name in names}) == len(names), "Case-colliding checkpoint members")
        for info in infos:
            name = info.filename
            _require(not name.startswith("/") and ".." not in name.split("/"), "Unsafe checkpoint member")
            _require(not stat.S_ISLNK(info.external_attr >> 16), "Checkpoint contains a symlink")
            _require(info.file_size < 16_000_000, "Checkpoint member too large")
        files = json.loads(archive.read("FILE_MANIFEST.json"))
        _require(set(files) == allowed - {"FILE_MANIFEST.json"}, "Checkpoint file manifest incomplete")
        for name, metadata in files.items():
            content = archive.read(name)
            _require(
                len(content) == metadata.get("bytes")
                and hashlib.sha256(content).hexdigest() == metadata.get("sha256"),
                f"Checkpoint member hash changed: {name}",
            )
        manifest = json.loads(archive.read("PRIVATE_CHECKPOINT_MANIFEST.json"))
        body = {key: value for key, value in manifest.items() if key != "checkpoint_manifest_sha256"}
        _require(manifest.get("schema") == "E45_PRIVATE_SHARD_CHECKPOINT_R4", "Wrong checkpoint schema")
        _require(manifest.get("checkpoint_manifest_sha256") == canonical_sha256(body), "Checkpoint manifest hash changed")
        identity = manifest.get("checkpoint_identity")
        _require(isinstance(identity, dict), "Checkpoint identity missing")
        _require(manifest.get("checkpoint_identity_sha256") == canonical_sha256(identity), "Checkpoint identity hash changed")
        states: list[dict[str, Any]] = []
        worker_hashes = manifest.get("worker_state_files")
        _require(isinstance(worker_hashes, dict), "Checkpoint worker hash map missing")
        _require(
            set(worker_hashes) == {Path(name).name for name in allowed if name.startswith("worker-states/")},
            "Checkpoint worker hash map has missing or extra entries",
        )
        for name in sorted(allowed):
            if not name.startswith("worker-states/"):
                continue
            content = archive.read(name)
            _require(worker_hashes.get(Path(name).name) == hashlib.sha256(content).hexdigest(), "Worker hash changed")
            state = json.loads(content)
            state_body = {key: value for key, value in state.items() if key != "state_sha256"}
            _require(state.get("state_sha256") == compute_json_sha256(state_body), "Worker state hash changed")
            _require(state.get("schema_version") == "E45_PRIVATE_P01_RESUME_STATE_R4", "Wrong worker state schema")
            _require(state.get("identity") == identity.get("generation_identity"), "Worker generation identity changed")
            states.append(state)
    return identity, states


def _collect_shard(shard_index: int, identity: dict[str, Any], states: list[dict[str, Any]], ids: list[str]) -> tuple[dict[str, str], dict[str, Any]]:
    """Lấy mọi câu đã lưu; ưu tiên kết quả sinh lại nếu đã hoàn tất."""
    expected_global = list(range(shard_index, len(ids), 4))
    expected_ids = [ids[index] for index in expected_global]
    _require(identity.get("shard_index") == shard_index and identity.get("shard_count") == 4, "Wrong shard checkpoint")
    _require(identity.get("global_question_ids_sha256") == hashlib.sha256("\n".join(expected_ids).encode()).hexdigest(), "Shard QID identity changed")
    completed: dict[int, dict[str, Any]] = {}
    due: set[int] = set()
    done: set[int] = set()
    for state in states:
        assigned = state.get("assigned_indices")
        finished = state.get("completed")
        records = state.get("records")
        _require(isinstance(assigned, list) and isinstance(finished, list) and isinstance(records, dict), "Malformed worker state")
        _require(len(assigned) == len(set(assigned)) and len(finished) == len(set(finished)), "Duplicate worker index")
        _require(set(assigned) == set(finished) == {int(key) for key in records}, "First-pass worker coverage incomplete")
        second_due = state.get("second_pass_indices")
        second_done = state.get("second_pass_completed")
        _require(isinstance(second_due, list) and isinstance(second_done, list), "Malformed second-pass state")
        _require(len(second_due) == len(set(second_due)) and len(second_done) == len(set(second_done)), "Duplicate second-pass index")
        _require(set(second_done).issubset(set(second_due).intersection(set(finished))), "Second-pass coverage invalid")
        due.update(second_due)
        done.update(second_done)
        for key, record in records.items():
            index = int(key)
            _require(str(index) == key and index not in completed and 0 <= index < len(expected_ids), "Duplicate/out-of-range shard record")
            _require(isinstance(record, dict) and isinstance(record.get("raw_answer"), str) and record["raw_answer"].strip(), "Empty shard answer")
            _require(record.get("finish_reason") in {"eos", "length", "other"}, "Invalid finish reason")
            _require(isinstance(record.get("prompt_input_ids_sha256"), str) and len(record["prompt_input_ids_sha256"]) == 64, "Prompt identity missing")
            if index in second_done:
                _require(record.get("initial_finish_reason") == "length" and isinstance(record.get("second_pass_tokens"), int), "Completed second pass invalid")
            elif index in second_due:
                _require(record.get("finish_reason") == "length" and record.get("first_pass_tokens") == 1024, "Fallback answer was not capped first pass")
            completed[index] = record
    _require(set(completed) == set(range(len(expected_ids))), "Shard lacks complete first-pass coverage")
    _require(len(due) == sum(len(s["second_pass_indices"]) for s in states), "Second-pass ownership overlaps")
    answers: dict[str, str] = {}
    for index, qid in enumerate(expected_ids):
        answer, _ = unified_clean(completed[index]["raw_answer"])
        _require(isinstance(answer, str) and answer.strip(), "Cleanup produced an empty answer")
        answers[qid] = answer
    report = {
        "shard_index": shard_index,
        "question_count": len(expected_ids),
        "second_pass_due": len(due),
        "second_pass_completed": len(done),
        "first_pass_fallback_count": len(due - done),
        "first_pass_fallback_qids": [expected_ids[index] for index in sorted(due - done)],
    }
    return answers, report


def merge(*, admission_path: Path, private_path: Path, candidate_path: Path, checkpoint_paths: list[Path], output_dir: Path) -> Path:
    """Đối chiếu bốn checkpoint rồi ghi submission ZIP sát deadline."""
    _require(len(checkpoint_paths) == 4, "Exactly four checkpoints are required")
    _require(not output_dir.exists(), "Emergency output directory already exists")
    admission = load_admission(admission_path)
    _, ids, private_identity = load_private_questions(private_path)
    source_order = list(json.loads(private_path.read_text(encoding="utf-8")))
    _require(set(source_order) == set(ids) and len(source_order) == len(ids), "Private source order is invalid")
    _require(admission["private_sha256"] == private_identity["private_sha256"], "Admission/private bytes differ")
    _require(admission["private_sample_ids_sha256"] == private_identity["sample_ids_sha256"], "Admission/private IDs differ")
    with tempfile.TemporaryDirectory(prefix="e45_emergency_candidate_") as temporary:
        materialize_candidate_adapter(
            candidate_archive=candidate_path, destination=Path(temporary) / "adapter", admission=admission,
        )
    answers: dict[str, str] = {}
    reports: list[dict[str, Any]] = []
    common: dict[str, Any] | None = None
    checkpoint_hashes: dict[str, str] = {}
    for shard_index, path in enumerate(checkpoint_paths):
        digest = file_sha256(path)
        identity, states = _read_checkpoint(path, digest)
        _require(identity.get("admission_sha256") == admission["admission_sha256"], "Checkpoint/admission differ")
        _require(identity.get("candidate_archive_sha256") == admission["candidate_archive_sha256"], "Checkpoint/candidate differ")
        _require(identity.get("candidate_adapter_sha256") == admission["candidate_adapter_sha256"], "Checkpoint/adapter differ")
        _require(identity.get("config_sha256") == admission["config_sha256"], "Checkpoint/config differ")
        _require(identity.get("private_sha256") == private_identity["private_sha256"], "Checkpoint/private differ")
        _require(identity.get("private_sample_ids_sha256") == private_identity["sample_ids_sha256"], "Checkpoint/private IDs differ")
        _require(identity.get("private_sample_size") == len(ids), "Checkpoint/private count differs")
        _, shard_identity = shard_questions(private_path=private_path, shard_index=shard_index, shard_count=4)
        _require(identity.get("shard_identity_sha256") == shard_identity["shard_identity_sha256"], "Checkpoint shard ownership differs")
        compared = {key: value for key, value in identity.items() if key not in {
            "shard_index", "shard_identity_sha256", "global_question_ids_sha256", "contexts_sha256",
            "context_manifest_sha256", "system_code_sha256", "private_scheduler_sha256",
        }}
        if common is None:
            common = compared
        _require(compared == common, "Checkpoint generation identities differ")
        shard_answers, shard_report = _collect_shard(shard_index, identity, states, ids)
        _require(not set(answers).intersection(shard_answers), "QID appears in multiple shards")
        answers.update(shard_answers)
        shard_report["checkpoint_sha256"] = digest
        shard_report["system_code_sha256"] = identity["system_code_sha256"]
        reports.append(shard_report)
        checkpoint_hashes[str(shard_index)] = digest
    _require(set(answers) == set(ids) and len(answers) == len(ids), "Merged private QID coverage is incomplete")
    submission = {qid: {"answer": answers[qid]} for qid in source_order}
    _require(all(set(row) == {"answer"} and isinstance(row["answer"], str) and row["answer"].strip() for row in submission.values()), "Invalid answer schema")
    payload = (json.dumps(submission, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    output_dir.mkdir(parents=True)
    (output_dir / "submission.json").write_bytes(payload)
    zip_path = output_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("submission.json", payload)
    with zipfile.ZipFile(zip_path) as archive:
        _require(archive.namelist() == ["submission.json"] and archive.read("submission.json") == payload, "Submission ZIP invalid")
    report = {
        "schema": "E45_PRIVATE_EMERGENCY_CHECKPOINT_MERGE_R1",
        "deviation": "unfinished_1536_restarts_use_saved_1024_first_pass",
        "private_reference_answers_read": False,
        "sample_size": len(ids),
        "sample_ids_sha256": private_identity["sample_ids_sha256"],
        "admission_sha256": admission["admission_sha256"],
        "candidate_archive_sha256": admission["candidate_archive_sha256"],
        "checkpoint_sha256s": checkpoint_hashes,
        "shards": reports,
        "first_pass_fallback_count": sum(row["first_pass_fallback_count"] for row in reports),
        "submission_sha256": hashlib.sha256(payload).hexdigest(),
        "submission_zip_sha256": file_sha256(zip_path),
    }
    (output_dir / "emergency-merge-report.json").write_bytes(_json_bytes(report))
    return zip_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--private-questions", type=Path, required=True)
    parser.add_argument("--candidate-bin", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True, help="Pass in shard 0, 1, 2, 3 order")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = merge(
        admission_path=args.admission, private_path=args.private_questions,
        candidate_path=args.candidate_bin, checkpoint_paths=args.checkpoint, output_dir=args.output,
    )
    print(json.dumps({"status": "READY_EMERGENCY_SUBMISSION", "path": str(path), "sha256": file_sha256(path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
