"""CPU fixtures for the private per-record checkpoint transport contract."""

from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

import pytest


PRIVATE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PRIVATE_ROOT / "src"))

from e45_private_checkpoint import (  # noqa: E402
    PrivateCheckpointError,
    find_private_resume_checkpoint,
    restore_private_checkpoint,
    verify_checkpoint_sidecar,
    write_private_checkpoint,
)
from e45_private_resumable_generation import (  # noqa: E402
    _make_state,
    _persist_state,
    _validate_state,
)
from uit_dsc_fixed_rag.e45_paired_generation import compute_json_sha256  # noqa: E402


def _context(index: int) -> dict[str, str | int]:
    body: dict[str, str | int] = {
        "sample_index": index,
        "question_id": f"q{index}",
        "question_sha256": "1" * 64,
        "evidence_sha256": "2" * 64,
        "prompt_sha256": "3" * 64,
        "prompt_input_ids_sha256": "4" * 64,
    }
    body["record_sha256"] = compute_json_sha256(body)
    return body


def _identity() -> dict[str, object]:
    return {
        "schema": "E45_PRIVATE_CHECKPOINT_IDENTITY_R4",
        "experiment_id": "E45-inference-aligned-parent-lora-v1",
        "admission_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "system_code_sha256": "c" * 64,
        "private_scheduler_sha256": "d" * 64,
        "candidate_archive_sha256": "e" * 64,
        "candidate_adapter_sha256": "f" * 64,
        "candidate_adapter_config_sha256": "0" * 64,
        "private_sha256": "9" * 64,
        "private_sample_ids_sha256": "8" * 64,
        "private_sample_size": 1918,
        "shard_index": 0,
        "shard_count": 4,
        "shard_identity_sha256": "7" * 64,
        "global_question_ids_sha256": "6" * 64,
        "contexts_sha256": "5" * 64,
        "context_manifest_sha256": "4" * 64,
        "generation_identity": {"identity_sha256": "3" * 64},
    }


def _write_valid_states(state_dir: Path, identity: dict[str, object]) -> dict[int, dict[str, str | int]]:
    contexts = {0: _context(0), 1: _context(1)}
    worker0 = _make_state(
        worker_path="replica_0", device="cuda:0", device_map=None,
        identity=identity, assigned_indices=[0],
    )
    worker0["completed"] = [0]
    worker0["records"] = {
        "0": {
            "raw_answer": "Cau tra loi", "finish_reason": "eos", "first_pass_tokens": 5,
            "prompt_input_ids_sha256": contexts[0]["prompt_input_ids_sha256"], "worker_path": "replica_0",
        }
    }
    worker0["status"] = "complete"
    worker1 = _make_state(
        worker_path="replica_1", device="cuda:1", device_map=None,
        identity=identity, assigned_indices=[1],
    )
    worker1["completed"] = [1]
    worker1["records"] = {
        "1": {
            "raw_answer": "Cau tra loi khac", "finish_reason": "length", "first_pass_tokens": 1024,
            "prompt_input_ids_sha256": contexts[1]["prompt_input_ids_sha256"], "worker_path": "replica_1",
        }
    }
    worker1["status"] = "complete"
    state_dir.mkdir(parents=True)
    _persist_state(state_dir / "worker-0-state.json", worker0)
    _persist_state(state_dir / "worker-1-state.json", worker1)
    _validate_state(
        state=worker0, expected_identity=identity, expected_assigned=[0], expected_worker_path="replica_0",
        expected_device="cuda:0", expected_device_map=None, contexts_by_index=contexts,
    )
    _validate_state(
        state=worker1, expected_identity=identity, expected_assigned=[1], expected_worker_path="replica_1",
        expected_device="cuda:1", expected_device_map=None, contexts_by_index=contexts,
    )
    return contexts


def test_checkpoint_round_trip_identity_and_state_hashes(tmp_path: Path) -> None:
    identity = _identity()
    source_states = tmp_path / "source-states"
    _write_valid_states(source_states, identity)
    archive = tmp_path / "E45_PRIVATE_SHARD_0_OF_4_CHECKPOINT.bin"
    written = write_private_checkpoint(
        archive_path=archive, checkpoint_identity=identity, state_dir=source_states, reason="fixture",
    )
    assert archive.is_file()
    assert written["worker_state_count"] == 2
    assert verify_checkpoint_sidecar(archive) == written["checkpoint_archive_sha256"]
    restored = tmp_path / "restored-states"
    restored_manifest = restore_private_checkpoint(
        archive_path=archive, destination_state_dir=restored, expected_identity=identity,
    )
    assert restored_manifest["checkpoint_identity"] == identity
    assert sorted(path.name for path in restored.glob("*.json")) == ["worker-0-state.json", "worker-1-state.json"]


def test_checkpoint_rejects_identity_sidecar_and_multiple_inputs(tmp_path: Path) -> None:
    identity = _identity()
    states = tmp_path / "states"
    _write_valid_states(states, identity)
    archive = tmp_path / "E45_PRIVATE_SHARD_0_OF_4_CHECKPOINT.bin"
    write_private_checkpoint(archive_path=archive, checkpoint_identity=identity, state_dir=states, reason="fixture")
    changed = dict(identity)
    changed["candidate_adapter_sha256"] = "x" * 64
    with pytest.raises(PrivateCheckpointError):
        restore_private_checkpoint(
            archive_path=archive, destination_state_dir=tmp_path / "wrong", expected_identity=changed,
        )
    sidecar = Path(str(archive) + ".sha256")
    sidecar.write_text("0" * 64 + "  altered.bin\n", encoding="ascii")
    with pytest.raises(PrivateCheckpointError):
        verify_checkpoint_sidecar(archive)

    # Restore the valid archive/sidecar into two distinct attached datasets;
    # discovery must reject the ambiguous resume rather than choose one.
    write_private_checkpoint(archive_path=archive, checkpoint_identity=identity, state_dir=states, reason="fixture-2")
    duplicate_root = tmp_path / "second"
    duplicate_root.mkdir()
    duplicate = duplicate_root / archive.name
    shutil.copy2(archive, duplicate)
    shutil.copy2(Path(str(archive) + ".sha256"), Path(str(duplicate) + ".sha256"))
    with pytest.raises(PrivateCheckpointError):
        find_private_resume_checkpoint(search_roots=[tmp_path], shard_index=0, shard_count=4)


def test_checkpoint_rejects_unsafe_archive_even_with_matching_sidecar(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.bin"
    with zipfile.ZipFile(archive, "w") as payload:
        payload.writestr("../escape.json", "{}")
    digest = __import__("hashlib").sha256(archive.read_bytes()).hexdigest()
    Path(str(archive) + ".sha256").write_text(f"{digest}  unsafe.bin\n", encoding="ascii")
    with pytest.raises(PrivateCheckpointError):
        restore_private_checkpoint(
            archive_path=archive, destination_state_dir=tmp_path / "unsafe-restore", expected_identity=_identity(),
        )
