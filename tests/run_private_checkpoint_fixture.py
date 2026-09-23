"""Run private-checkpoint corruption fixtures without pytest or GPU packages."""

from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e45_private_checkpoint import (  # noqa: E402
    PrivateCheckpointError,
    find_private_resume_checkpoint,
    restore_private_checkpoint,
    verify_checkpoint_sidecar,
    write_private_checkpoint,
)
import e45_private_resumable_generation as resume_module  # noqa: E402
from e45_private_resumable_generation import (  # noqa: E402
    _make_state,
    _persist_state,
    _validate_state,
    execute_resumable_private_p01_generation,
)
from uit_dsc_fixed_rag.e45_paired_generation import compute_json_sha256  # noqa: E402


def expect_raises(exc_type: type[BaseException], callback: object) -> None:
    try:
        callback()  # type: ignore[operator]
    except exc_type:
        return
    raise AssertionError(f"Expected {exc_type.__name__}")


def identity() -> dict[str, object]:
    return {
        "schema": "E45_PRIVATE_CHECKPOINT_IDENTITY_R4", "experiment_id": "E45-inference-aligned-parent-lora-v1",
        "admission_sha256": "a" * 64, "config_sha256": "b" * 64, "system_code_sha256": "c" * 64,
        "private_scheduler_sha256": "d" * 64, "candidate_archive_sha256": "e" * 64,
        "candidate_adapter_sha256": "f" * 64, "candidate_adapter_config_sha256": "0" * 64,
        "private_sha256": "9" * 64, "private_sample_ids_sha256": "8" * 64, "private_sample_size": 1918,
        "shard_index": 0, "shard_count": 4, "shard_identity_sha256": "7" * 64,
        "global_question_ids_sha256": "6" * 64, "contexts_sha256": "5" * 64,
        "context_manifest_sha256": "4" * 64, "generation_identity": {"identity_sha256": "3" * 64},
    }


def context(index: int) -> dict[str, str | int]:
    item: dict[str, str | int] = {
        "sample_index": index, "question_id": f"q{index}", "question_sha256": "1" * 64,
        "evidence_sha256": "2" * 64, "prompt_sha256": "3" * 64,
        "prompt_input_ids_sha256": "4" * 64,
    }
    item["record_sha256"] = compute_json_sha256(item)
    return item


def write_states(state_dir: Path, run_identity: dict[str, object]) -> None:
    contexts = {0: context(0), 1: context(1)}
    for index, worker_path, device, filename, answer, finish in (
        (0, "replica_0", "cuda:0", "worker-0-state.json", "Cau tra loi", "eos"),
        (1, "replica_1", "cuda:1", "worker-1-state.json", "Cau khac", "length"),
    ):
        state = _make_state(
            worker_path=worker_path, device=device, device_map=None,
            identity=run_identity, assigned_indices=[index],
        )
        state["completed"] = [index]
        state["records"] = {
            str(index): {
                "raw_answer": answer, "finish_reason": finish, "first_pass_tokens": 5,
                "prompt_input_ids_sha256": contexts[index]["prompt_input_ids_sha256"], "worker_path": worker_path,
            }
        }
        state["status"] = "complete"
        state_dir.mkdir(parents=True, exist_ok=True)
        _persist_state(state_dir / filename, state)
        _validate_state(
            state=state, expected_identity=run_identity, expected_assigned=[index],
            expected_worker_path=worker_path, expected_device=device, expected_device_map=None,
            contexts_by_index=contexts,
        )


class FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return False

    @staticmethod
    def empty_cache() -> None:
        return None


class FakeTorch:
    cuda = FakeCuda()


class FakeWorker:
    """CPU fixture that exercises state/restart logic without model loading."""

    base_model_path = "fake-base"
    tokenizer_path = Path("fake-tokenizer")
    adapter_path = Path("fake-adapter")
    config_raw: dict[str, object] = {}
    config = SimpleNamespace(path=Path("fake-config.json"))
    config_sha256 = "a" * 64
    inference_cfg = {
        "replica_max_input_tokens": 0,
        "initial_max_new_tokens": 1024,
        "second_pass_max_new_tokens": 1536,
    }

    def __init__(self, run_identity: dict[str, object], hashes: dict[str, str]):
        self._identity = run_identity
        self._hashes = hashes
        self.calls: list[str] = []

    def verify_runtime(self) -> dict[str, str]:
        return {}

    def identity(self) -> dict[str, object]:
        return self._identity

    def _tokenizer(self) -> object:
        return object()

    def _load_model(self, *_: object, **__: object) -> object:
        return object()

    def generate_single(self, _model: object, _tokenizer: object, prompt: str, _limit: int) -> tuple[str, str, int, str]:
        self.calls.append(prompt)
        return f"answer-{prompt}", "eos", 3, self._hashes[prompt]


def resumable_context(index: int) -> dict[str, str | int | bool]:
    item: dict[str, str | int | bool] = {
        "sample_index": index, "question_id": f"q{index}", "question_sha256": "1" * 64,
        "evidence_sha256": "2" * 64, "prompt_sha256": "3" * 64,
        "prompt_input_ids_sha256": hashlib.sha256(f"input-{index}".encode()).hexdigest(),
        "prompt": f"prompt-{index}", "input_tokens": 1, "answers_used": False,
    }
    item["record_sha256"] = compute_json_sha256(item)
    return item


def main() -> int:
    # The runtime module imports torch lazily.  Inject a small CPU fixture so
    # the scheduler can be tested without installing a model/GPU stack.
    sys.modules["torch"] = FakeTorch()
    with tempfile.TemporaryDirectory(prefix="e45_private_checkpoint_fixture_") as temporary:
        root = Path(temporary)
        run_identity = identity()
        states = root / "states"
        write_states(states, run_identity)
        archive = root / "E45_PRIVATE_SHARD_0_OF_4_CHECKPOINT.bin"
        written = write_private_checkpoint(
            archive_path=archive, checkpoint_identity=run_identity, state_dir=states, reason="fixture",
        )
        assert verify_checkpoint_sidecar(archive) == written["checkpoint_archive_sha256"]
        restored = root / "restored"
        manifest = restore_private_checkpoint(
            archive_path=archive, destination_state_dir=restored, expected_identity=run_identity,
        )
        assert manifest["checkpoint_identity"] == run_identity
        assert {path.name for path in restored.glob("*.json")} == {"worker-0-state.json", "worker-1-state.json"}

        altered = dict(run_identity)
        altered["candidate_adapter_sha256"] = "x" * 64
        expect_raises(
            PrivateCheckpointError,
            lambda: restore_private_checkpoint(
                archive_path=archive, destination_state_dir=root / "wrong", expected_identity=altered,
            ),
        )

        # A controlled wall-clock stop must retain first-pass state, then a
        # fresh output directory must restore it and finish only missing rows.
        scheduler_identity = {
            "experiment_id": "E45-inference-aligned-parent-lora-v1",
            "config_sha256": "b" * 64,
            "base_model": "fake-base", "base_revision": "fake-revision",
            "tokenizer_sha256": "c" * 64, "adapter_sha256": "d" * 64,
            "decoding_sha256": "e" * 64, "cleanup_sha256": "f" * 64,
            "identity_sha256": "0" * 64,
        }
        contexts = [resumable_context(index) for index in range(2)]
        worker = FakeWorker(scheduler_identity, {row["prompt"]: row["prompt_input_ids_sha256"] for row in contexts})
        checkpoint_identity = dict(run_identity)
        checkpoint_identity["generation_identity"] = scheduler_identity
        checkpoint_identity["shard_index"] = 1
        checkpoint = root / "E45_PRIVATE_SHARD_1_OF_4_CHECKPOINT.bin"
        first_output = root / "first-output" / "generation"
        clock = iter((0.0, 2.0))  # first QID commits; second reaches safety stop
        original_time = resume_module.time.time
        resume_module.time.time = lambda: next(clock)
        try:
            first = execute_resumable_private_p01_generation(
                worker=worker, contexts=contexts, output_dir=first_output, arm_name="private-candidate",
                checkpoint_archive=checkpoint, checkpoint_identity=checkpoint_identity, deadline_epoch=1.0,
            )
        finally:
            resume_module.time.time = original_time
        assert not first.completed and checkpoint.is_file()
        assert worker.calls == ["prompt-0"]
        second_output = root / "second-output" / "generation"
        restore_private_checkpoint(
            archive_path=checkpoint, destination_state_dir=second_output / "worker-states",
            expected_identity=checkpoint_identity,
        )
        resumed_worker = FakeWorker(scheduler_identity, {row["prompt"]: row["prompt_input_ids_sha256"] for row in contexts})
        second = execute_resumable_private_p01_generation(
            worker=resumed_worker, contexts=contexts, output_dir=second_output, arm_name="private-candidate",
            checkpoint_archive=checkpoint, checkpoint_identity=checkpoint_identity, deadline_epoch=None,
        )
        assert second.completed and second.raw_path is not None and second.clean_path is not None
        assert resumed_worker.calls == ["prompt-1"]
        assert len(second.raw_path.read_text(encoding="utf-8").splitlines()) == 2
        Path(str(archive) + ".sha256").write_text("0" * 64 + "  altered.bin\n", encoding="ascii")
        expect_raises(PrivateCheckpointError, lambda: verify_checkpoint_sidecar(archive))

        # Reissue valid bytes then prove an ambiguous pair is rejected.
        write_private_checkpoint(archive_path=archive, checkpoint_identity=run_identity, state_dir=states, reason="fixture-2")
        duplicate_dir = root / "duplicate"
        duplicate_dir.mkdir()
        duplicate = duplicate_dir / archive.name
        shutil.copy2(archive, duplicate)
        shutil.copy2(Path(str(archive) + ".sha256"), Path(str(duplicate) + ".sha256"))
        expect_raises(
            PrivateCheckpointError,
            lambda: find_private_resume_checkpoint(search_roots=[root], shard_index=0, shard_count=4),
        )

        # A checkpoint from a different shard must not be silently ignored.
        shutil.rmtree(duplicate_dir)
        other_identity = dict(run_identity)
        other_identity["shard_index"] = 1
        other = root / "E45_PRIVATE_SHARD_1_OF_4_CHECKPOINT.bin"
        write_private_checkpoint(archive_path=other, checkpoint_identity=other_identity, state_dir=states, reason="other")
        expect_raises(
            PrivateCheckpointError,
            lambda: find_private_resume_checkpoint(search_roots=[root], shard_index=0, shard_count=4),
        )
        other.unlink()
        Path(str(other) + ".sha256").unlink()

        # A finished bundle for this same shard is not a resume archive.  A
        # fresh notebook session must fail rather than regenerate completed
        # private answers from an accidentally attached final output dataset.
        completed_bundle = root / "E45_PRIVATE_SHARD_0_OF_4.bin"
        with zipfile.ZipFile(completed_bundle, "w") as payload:
            payload.writestr(
                "shard-report.json",
                '{"shard_identity":{"shard_index":0,"shard_count":4}}',
            )
        expect_raises(
            PrivateCheckpointError,
            lambda: find_private_resume_checkpoint(search_roots=[root], shard_index=0, shard_count=4),
        )
        completed_bundle.unlink()

        unsafe = root / "unsafe.bin"
        with zipfile.ZipFile(unsafe, "w") as payload:
            payload.writestr("../escape.json", "{}")
        unsafe_hash = hashlib.sha256(unsafe.read_bytes()).hexdigest()
        Path(str(unsafe) + ".sha256").write_text(f"{unsafe_hash}  unsafe.bin\n", encoding="ascii")
        expect_raises(
            PrivateCheckpointError,
            lambda: restore_private_checkpoint(
                archive_path=unsafe, destination_state_dir=root / "unsafe-restore", expected_identity=run_identity,
            ),
        )
    print("private checkpoint fixture: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
