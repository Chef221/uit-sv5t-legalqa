"""Resumable private execution of the frozen E45 P01 generation policy.

This module does not alter the E45 authority.  It uses the authority's model
loader, tokenizer call, greedy generation arguments, finish classifier and
cleanup function.  Its only addition is an atomic state snapshot after each
committed answer so a Kaggle session can stop at a safe boundary and resume
only the unanswered QIDs.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from e45_private_checkpoint import PrivateCheckpointError, write_private_checkpoint
from uit_dsc_fixed_rag.corpus import file_sha256
from uit_dsc_fixed_rag.e45_paired_generation import (
    E45GeneratorWorker,
    GenerationError,
    _atomic_json,
    _atomic_jsonl,
    _context_identity,
    _is_cuda_oom,
    compute_json_sha256,
)
from uit_dsc_fixed_rag.e45_parent_training import E45Config
from uit_dsc_fixed_rag.final_private_p01 import unified_clean


PRIVATE_RESUME_STATE_SCHEMA = "E45_PRIVATE_P01_RESUME_STATE_R4"
_REPLICA_STATE_NAMES = ("worker-0-state.json", "worker-1-state.json")
_SHARDED_STATE_NAME = "sharded-worker-state.json"


class PrivateGenerationResumeError(RuntimeError):
    """Raised when a resumable P01 state loses an immutable binding."""


@dataclass(frozen=True)
class PrivateGenerationOutcome:
    """Result from one session segment of private generation."""

    completed: bool
    checkpoint: dict[str, Any]
    completed_first_pass: int
    total: int
    raw_path: Path | None = None
    clean_path: Path | None = None
    report: dict[str, Any] | None = None
    reason: str = ""


def _state_body(state: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in state.items() if key != "state_sha256"}


def _persist_state(path: Path, state: dict[str, Any]) -> None:
    state["state_sha256"] = compute_json_sha256(_state_body(state))
    _atomic_json(path, state)


def _read_state(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrivateGenerationResumeError(f"Invalid private worker state: {path.name}") from exc
    if not isinstance(value, dict):
        raise PrivateGenerationResumeError(f"Private worker state is not an object: {path.name}")
    return value


def _make_state(
    *, worker_path: str, device: str | None, device_map: str | None,
    identity: dict[str, Any], assigned_indices: list[int],
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "schema_version": PRIVATE_RESUME_STATE_SCHEMA,
        "worker_path": worker_path,
        "identity": identity,
        "assigned_indices": assigned_indices,
        "completed": [],
        "records": {},
        "status": "ready",
        "second_pass_indices": [],
        "second_pass_completed": [],
    }
    if device is not None:
        state["device"] = device
    if device_map is not None:
        state["device_map"] = device_map
    if not assigned_indices:
        state["status"] = "complete"
    return state


def _parse_record_index(key: str) -> int:
    try:
        index = int(key)
    except (TypeError, ValueError) as exc:
        raise PrivateGenerationResumeError("Worker state record index is invalid") from exc
    if str(index) != str(key):
        raise PrivateGenerationResumeError("Worker state record index is not canonical")
    return index


def _validate_state(
    *,
    state: dict[str, Any],
    expected_identity: dict[str, Any],
    expected_assigned: list[int],
    expected_worker_path: str,
    expected_device: str | None,
    expected_device_map: str | None,
    contexts_by_index: dict[int, dict[str, Any]],
) -> None:
    """Validate all persisted record state before it can affect a resumed arm."""
    allowed = {
        "schema_version", "worker_path", "identity", "assigned_indices", "completed", "records",
        "status", "second_pass_indices", "second_pass_completed", "state_sha256", "device",
        "device_map", "stop_reason", "fallback_reason", "raw_record_sha256s", "second_pass_device_map",
    }
    required = {
        "schema_version", "worker_path", "identity", "assigned_indices", "completed", "records",
        "status", "second_pass_indices", "second_pass_completed", "state_sha256",
    }
    if set(state) - allowed or not required.issubset(state):
        raise PrivateGenerationResumeError("Worker state schema has unexpected or missing fields")
    if state["state_sha256"] != compute_json_sha256(_state_body(state)):
        raise PrivateGenerationResumeError("Worker state hash is invalid")
    if (
        state["schema_version"] != PRIVATE_RESUME_STATE_SCHEMA
        or state["worker_path"] != expected_worker_path
        or state["identity"] != expected_identity
        or state["assigned_indices"] != expected_assigned
    ):
        raise PrivateGenerationResumeError("Worker state immutable identity changed")
    if expected_device is None:
        if state.get("device_map") != expected_device_map or "device" in state:
            raise PrivateGenerationResumeError("Sharded worker device map changed")
    elif state.get("device") != expected_device or "device_map" in state:
        raise PrivateGenerationResumeError("Replica worker device changed")
    if state["status"] not in {"ready", "running", "complete", "oom", "stopped"}:
        raise PrivateGenerationResumeError("Worker state status is unsafe for resume")
    assigned = state["assigned_indices"]
    completed = state["completed"]
    if (
        not all(isinstance(value, int) for value in assigned + completed)
        or len(assigned) != len(set(assigned))
        or len(completed) != len(set(completed))
        or not set(completed).issubset(set(assigned))
    ):
        raise PrivateGenerationResumeError("Worker state completion indexes are invalid")
    record_indexes = {_parse_record_index(key) for key in state["records"]}
    if record_indexes != set(completed):
        raise PrivateGenerationResumeError("Worker state records do not match completed indexes")
    if state["status"] == "complete" and set(completed) != set(assigned):
        raise PrivateGenerationResumeError("Complete worker state lacks assigned records")
    if state["status"] == "stopped" and not isinstance(state.get("stop_reason"), str):
        raise PrivateGenerationResumeError("Stopped worker state lacks a stop reason")
    for index in completed:
        record = state["records"].get(str(index))
        if not isinstance(record, dict) or set(record) - {
            "raw_answer", "finish_reason", "first_pass_tokens", "prompt_input_ids_sha256",
            "worker_path", "fallback_reason", "initial_finish_reason", "second_pass_tokens",
        }:
            raise PrivateGenerationResumeError("Worker state record schema changed")
        context = contexts_by_index.get(index)
        if context is None:
            raise PrivateGenerationResumeError("Worker state record falls outside prepared contexts")
        if (
            not isinstance(record.get("raw_answer"), str)
            or not record["raw_answer"].strip()
            or record.get("finish_reason") not in {"eos", "length", "other"}
            or not isinstance(record.get("first_pass_tokens"), int)
            or record["first_pass_tokens"] < 0
            or record.get("prompt_input_ids_sha256") != context["prompt_input_ids_sha256"]
            or not isinstance(record.get("worker_path"), str)
        ):
            raise PrivateGenerationResumeError("Worker state record is not bound to its P01 context")
        if "initial_finish_reason" in record:
            if (
                record["initial_finish_reason"] != "length"
                or not isinstance(record.get("second_pass_tokens"), int)
                or record["second_pass_tokens"] < 0
            ):
                raise PrivateGenerationResumeError("Worker second-pass state is invalid")
    second_indices = state["second_pass_indices"]
    second_completed = state["second_pass_completed"]
    if (
        not all(isinstance(value, int) for value in second_indices + second_completed)
        or len(second_indices) != len(set(second_indices))
        or len(second_completed) != len(set(second_completed))
        or not set(second_indices).issubset(set(completed))
        or not set(second_completed).issubset(set(second_indices))
    ):
        raise PrivateGenerationResumeError("Worker second-pass indexes are invalid")
    for index in second_completed:
        if state["records"][str(index)].get("initial_finish_reason") != "length":
            raise PrivateGenerationResumeError("Worker second-pass completion is unbound")
    raw_hashes = state.get("raw_record_sha256s")
    if raw_hashes is not None and (
        not isinstance(raw_hashes, list)
        or not all(isinstance(value, str) and len(value) == 64 for value in raw_hashes)
        or len(raw_hashes) != len(set(raw_hashes))
    ):
        raise PrivateGenerationResumeError("Worker raw-record hash binding is invalid")


def _snapshot(
    *, checkpoint_spec: dict[str, Any], checkpoint_lock: Any, states_dir: Path, reason: str,
) -> dict[str, Any]:
    try:
        with checkpoint_lock:
            return write_private_checkpoint(
                archive_path=Path(checkpoint_spec["archive_path"]),
                checkpoint_identity=checkpoint_spec["identity"],
                state_dir=states_dir,
                reason=reason,
            )
    except PrivateCheckpointError as exc:
        raise PrivateGenerationResumeError("Cannot persist private checkpoint") from exc


def _worker_from_spec(worker_spec: dict[str, Any]) -> E45GeneratorWorker:
    return E45GeneratorWorker(
        base_model_path=worker_spec["base_model_path"],
        tokenizer_path=Path(worker_spec["tokenizer_path"]),
        adapter_path=Path(worker_spec["adapter_path"]),
        config=E45Config(
            raw=worker_spec["config_raw"],
            path=Path(worker_spec["config_path"]),
            sha256=worker_spec["config_sha256"],
        ),
    )


def _deadline_reached(deadline_epoch: float | None) -> bool:
    return deadline_epoch is not None and time.time() >= deadline_epoch


def _replica_process(
    *, worker_spec: dict[str, Any], contexts: list[dict[str, Any]], device: str,
    state_path: str, checkpoint_spec: dict[str, Any], checkpoint_lock: Any,
    deadline_epoch: float | None,
) -> None:
    """Run only unfinished records of one frozen P01 short-prompt partition."""
    path = Path(state_path)
    states_dir = path.parent
    state = _read_state(path)
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.set_device(int(device.rsplit(":", 1)[-1]))
        state["status"] = "running"
        state.pop("stop_reason", None)
        _persist_state(path, state)
        _snapshot(
            checkpoint_spec=checkpoint_spec, checkpoint_lock=checkpoint_lock,
            states_dir=states_dir, reason=f"{state['worker_path']}:started",
        )
        worker = _worker_from_spec(worker_spec)
        worker.verify_runtime()
        tokenizer = worker._tokenizer()
        model = worker._load_model(device)
        for prepared in contexts:
            index = prepared["sample_index"]
            if index in state["completed"]:
                continue
            if _deadline_reached(deadline_epoch):
                state.update({"status": "stopped", "stop_reason": "wall_clock_safety_margin"})
                _persist_state(path, state)
                _snapshot(
                    checkpoint_spec=checkpoint_spec, checkpoint_lock=checkpoint_lock,
                    states_dir=states_dir, reason=f"{state['worker_path']}:wall-clock-stop",
                )
                return
            answer, finish, count, input_hash = worker.generate_single(
                model, tokenizer, prepared["prompt"], worker.inference_cfg["initial_max_new_tokens"]
            )
            if input_hash != prepared["prompt_input_ids_sha256"]:
                raise PrivateGenerationResumeError("Replica P01 input IDs differ from prepared context")
            state["records"][str(index)] = {
                "raw_answer": answer,
                "finish_reason": finish,
                "first_pass_tokens": count,
                "prompt_input_ids_sha256": input_hash,
                "worker_path": state["worker_path"],
            }
            state["completed"].append(index)
            _persist_state(path, state)
            _snapshot(
                checkpoint_spec=checkpoint_spec, checkpoint_lock=checkpoint_lock,
                states_dir=states_dir, reason=f"{state['worker_path']}:record-{index}",
            )
        state["status"] = "complete"
        _persist_state(path, state)
        _snapshot(
            checkpoint_spec=checkpoint_spec, checkpoint_lock=checkpoint_lock,
            states_dir=states_dir, reason=f"{state['worker_path']}:first-pass-complete",
        )
    except Exception as exc:
        if _is_cuda_oom(exc):
            state.update({"status": "oom", "fallback_reason": "cuda_out_of_memory"})
            _persist_state(path, state)
            _snapshot(
                checkpoint_spec=checkpoint_spec, checkpoint_lock=checkpoint_lock,
                states_dir=states_dir, reason=f"{state['worker_path']}:cuda-oom",
            )
            return
        state.update({
            "status": "failed", "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(limit=10),
        })
        # A failed state is intentionally not snapshotted: a later session must
        # never turn an arbitrary exception into an approved fallback.
        _persist_state(path, state)
        raise


def _first_pass_results(states: dict[str, dict[str, Any]]) -> dict[int, tuple[str, dict[str, Any]]]:
    results: dict[int, tuple[str, dict[str, Any]]] = {}
    for name, state in states.items():
        for index in state["completed"]:
            if index in results:
                raise PrivateGenerationResumeError("Two worker states own the same private record")
            results[index] = (name, state["records"][str(index)])
    return results


def _refresh_states(
    *, states_dir: Path, identity: dict[str, Any], assignments: dict[str, list[int]],
    contexts_by_index: dict[int, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Reload and validate the current durable states after a worker boundary."""
    specs = {
        "worker-0": ("replica_0", "cuda:0", None, _REPLICA_STATE_NAMES[0]),
        "worker-1": ("replica_1", "cuda:1", None, _REPLICA_STATE_NAMES[1]),
    }
    if "sharded-worker" in assignments:
        specs["sharded-worker"] = ("sharded", None, "balanced", _SHARDED_STATE_NAME)
    observed_names = {path.name for path in states_dir.glob("*.json")}
    required_names = {spec[3] for spec in specs.values()}
    if observed_names != required_names:
        raise PrivateGenerationResumeError("Private worker-state set is missing, stale, or mixed")
    states: dict[str, dict[str, Any]] = {}
    for name, (worker_path, device, device_map, filename) in specs.items():
        state = _read_state(states_dir / filename)
        _validate_state(
            state=state, expected_identity=identity, expected_assigned=assignments[name],
            expected_worker_path=worker_path, expected_device=device, expected_device_map=device_map,
            contexts_by_index=contexts_by_index,
        )
        states[name] = state
    return states


def _write_initial_states(
    *, states_dir: Path, identity: dict[str, Any], assignments: dict[str, list[int]],
    contexts_by_index: dict[int, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if states_dir.exists() and any(states_dir.iterdir()):
        # A restored checkpoint may already contain a sharded state.  Its
        # assignment depends on whether the replicas previously OOM'd, so it
        # cannot be validated until after the replica states are read below.
        observed = {path.name for path in states_dir.glob("*.json")}
        allowed = set(_REPLICA_STATE_NAMES) | {_SHARDED_STATE_NAME}
        if not set(_REPLICA_STATE_NAMES).issubset(observed) or observed - allowed:
            raise PrivateGenerationResumeError("Private resume state file set is missing, stale, or mixed")
        states: dict[str, dict[str, Any]] = {}
        for rank, name in enumerate(("worker-0", "worker-1")):
            state = _read_state(states_dir / _REPLICA_STATE_NAMES[rank])
            _validate_state(
                state=state, expected_identity=identity, expected_assigned=assignments[name],
                expected_worker_path=f"replica_{rank}", expected_device=f"cuda:{rank}",
                expected_device_map=None, contexts_by_index=contexts_by_index,
            )
            states[name] = state
        return states
    states_dir.mkdir(parents=True, exist_ok=True)
    first = _make_state(
        worker_path="replica_0", device="cuda:0", device_map=None,
        identity=identity, assigned_indices=assignments["worker-0"],
    )
    second = _make_state(
        worker_path="replica_1", device="cuda:1", device_map=None,
        identity=identity, assigned_indices=assignments["worker-1"],
    )
    _persist_state(states_dir / _REPLICA_STATE_NAMES[0], first)
    _persist_state(states_dir / _REPLICA_STATE_NAMES[1], second)
    return {"worker-0": first, "worker-1": second}


def _run_sharded_first_pass(
    *, worker: E45GeneratorWorker, state: dict[str, Any], state_path: Path,
    rows: list[dict[str, Any]], long_indexes: set[int], checkpoint_spec: dict[str, Any],
    checkpoint_lock: Any, deadline_epoch: float | None,
) -> bool:
    """Run or resume the exact P01 two-T4 fallback; return False at safe stop."""
    import torch

    state["status"] = "running"
    state.pop("stop_reason", None)
    _persist_state(state_path, state)
    _snapshot(
        checkpoint_spec=checkpoint_spec, checkpoint_lock=checkpoint_lock,
        states_dir=state_path.parent, reason="sharded:first-pass-started",
    )
    model: Any | None = None
    try:
        tokenizer = worker._tokenizer()
        model = worker._load_model(None, sharded=True)
        for prepared in rows:
            index = prepared["sample_index"]
            if index in state["completed"]:
                continue
            if _deadline_reached(deadline_epoch):
                state.update({"status": "stopped", "stop_reason": "wall_clock_safety_margin"})
                _persist_state(state_path, state)
                _snapshot(
                    checkpoint_spec=checkpoint_spec, checkpoint_lock=checkpoint_lock,
                    states_dir=state_path.parent, reason="sharded:wall-clock-stop",
                )
                return False
            answer, finish, count, input_hash = worker.generate_single(
                model, tokenizer, prepared["prompt"], worker.inference_cfg["initial_max_new_tokens"]
            )
            if input_hash != prepared["prompt_input_ids_sha256"]:
                raise PrivateGenerationResumeError("Sharded P01 input IDs differ from prepared context")
            state["records"][str(index)] = {
                "raw_answer": answer,
                "finish_reason": finish,
                "first_pass_tokens": count,
                "prompt_input_ids_sha256": input_hash,
                "worker_path": "sharded",
                "fallback_reason": "long_prompt" if index in long_indexes else "replica_cuda_out_of_memory",
            }
            state["completed"].append(index)
            _persist_state(state_path, state)
            _snapshot(
                checkpoint_spec=checkpoint_spec, checkpoint_lock=checkpoint_lock,
                states_dir=state_path.parent, reason=f"sharded:record-{index}",
            )
        state["status"] = "complete"
        _persist_state(state_path, state)
        _snapshot(
            checkpoint_spec=checkpoint_spec, checkpoint_lock=checkpoint_lock,
            states_dir=state_path.parent, reason="sharded:first-pass-complete",
        )
        return True
    except Exception as exc:
        state.update({
            "status": "failed", "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(limit=10),
        })
        _persist_state(state_path, state)
        raise PrivateGenerationResumeError("P01 sharded fallback failed; it must not silently retry") from exc
    finally:
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _finalize_records(
    *, contexts: list[dict[str, Any]], states: dict[str, dict[str, Any]],
    output_dir: Path, arm_name: str, identity: dict[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    provisional = _first_pass_results(states)
    if set(provisional) != set(range(len(contexts))):
        raise PrivateGenerationResumeError("Cannot finalize private generation with incomplete first-pass coverage")
    raw_rows: list[dict[str, Any]] = []
    clean_rows: list[dict[str, Any]] = []
    for index, prepared in enumerate(contexts):
        _, result = provisional[index]
        if result["prompt_input_ids_sha256"] != prepared["prompt_input_ids_sha256"]:
            raise PrivateGenerationResumeError("Persisted P01 input IDs differ from prepared context")
        raw = {
            "sample_index": index,
            "question_id": prepared["question_id"],
            **_context_identity(prepared),
            **identity,
            "worker_path": result["worker_path"],
            "raw_answer": result["raw_answer"],
            "initial_finish_reason": result.get("initial_finish_reason", result["finish_reason"]),
            "finish_reason": result["finish_reason"],
            "first_pass_tokens": result["first_pass_tokens"],
            "second_pass_tokens": result.get("second_pass_tokens", 0),
            "fallback_reason": result.get("fallback_reason", "none"),
        }
        raw["record_sha256"] = compute_json_sha256(raw)
        raw_rows.append(raw)
        answer, trim = unified_clean(raw["raw_answer"])
        clean = {
            "sample_index": index,
            "question_id": raw["question_id"],
            "raw_record_sha256": raw["record_sha256"],
            "cleanup_sha256": identity["cleanup_sha256"],
            "clean_answer": answer,
            "cleanup_report": trim,
        }
        clean["record_sha256"] = compute_json_sha256(clean)
        clean_rows.append(clean)
    raw_path = output_dir / "raw-records.jsonl"
    clean_path = output_dir / "clean-records.jsonl"
    _atomic_jsonl(raw_path, raw_rows)
    _atomic_jsonl(clean_path, clean_rows)
    for name, state in states.items():
        state["raw_record_sha256s"] = [raw_rows[index]["record_sha256"] for index in state["completed"]]
        _persist_state(
            output_dir / "worker-states" / (
                _SHARDED_STATE_NAME if name == "sharded-worker" else f"{name}-state.json"
            ),
            state,
        )
    report = {
        "schema_version": "1.0",
        "experiment_id": identity["experiment_id"],
        "arm_name": arm_name,
        "identity": identity,
        "raw_records_sha256": file_sha256(raw_path),
        "clean_records_sha256": file_sha256(clean_path),
        "sample_size": len(raw_rows),
        "initial_length_finish_count": sum(row["initial_finish_reason"] == "length" for row in raw_rows),
        "final_length_finish_count": sum(row["finish_reason"] == "length" for row in raw_rows),
        "second_pass_count": sum(row["initial_finish_reason"] == "length" for row in raw_rows),
        "worker_state_files": sorted(path.name for path in (output_dir / "worker-states").glob("*.json")),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(output_dir / "generation-report.json", report)
    return raw_path, clean_path, report


def _checkpoint_outcome(
    *, checkpoint_spec: dict[str, Any], checkpoint_lock: Any, states_dir: Path,
    states: dict[str, dict[str, Any]], total: int, reason: str,
) -> PrivateGenerationOutcome:
    checkpoint = _snapshot(
        checkpoint_spec=checkpoint_spec, checkpoint_lock=checkpoint_lock,
        states_dir=states_dir, reason=reason,
    )
    completed = len(_first_pass_results(states))
    return PrivateGenerationOutcome(
        completed=False, checkpoint=checkpoint, completed_first_pass=completed,
        total=total, reason=reason,
    )


def execute_resumable_private_p01_generation(
    *, worker: E45GeneratorWorker, contexts: list[dict[str, Any]], output_dir: Path,
    arm_name: str, checkpoint_archive: Path, checkpoint_identity: dict[str, Any],
    deadline_epoch: float | None,
) -> PrivateGenerationOutcome:
    """Run the P01 policy with per-answer checkpointing and strict resume state.

    ``output_dir/worker-states`` may contain a state directory restored only by
    :func:`e45_private_checkpoint.restore_private_checkpoint`.  Any other
    pre-existing generation output is rejected.
    """
    import torch

    if not contexts:
        raise PrivateGenerationResumeError("Cannot generate an empty private shard")
    worker.verify_runtime()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    permitted = {"worker-states"}
    if any(path.name not in permitted for path in output_dir.iterdir()):
        raise PrivateGenerationResumeError("Private generation output contains stale non-state files")
    if torch.cuda.is_available() and torch.cuda.device_count() < 2:
        raise PrivateGenerationResumeError("Frozen private P01 execution requires two visible GPUs")
    for index, row in enumerate(contexts):
        if row.get("sample_index") != index:
            raise PrivateGenerationResumeError("Private contexts must preserve exact sample-index order")
        _context_identity(row)
    identity = worker.identity()
    if checkpoint_identity.get("generation_identity") != identity:
        raise PrivateGenerationResumeError("Private checkpoint identity does not bind this worker")
    contexts_by_index = {row["sample_index"]: row for row in contexts}
    threshold = worker.inference_cfg["replica_max_input_tokens"]
    short = [row for row in contexts if row["input_tokens"] <= threshold]
    long = [row for row in contexts if row["input_tokens"] > threshold]
    assignments: dict[str, list[int]] = {
        "worker-0": [row["sample_index"] for row in short[0::2]],
        "worker-1": [row["sample_index"] for row in short[1::2]],
    }
    states_dir = output_dir / "worker-states"
    states = _write_initial_states(
        states_dir=states_dir, identity=identity, assignments=assignments,
        contexts_by_index=contexts_by_index,
    )
    checkpoint_spec = {"archive_path": str(checkpoint_archive), "identity": checkpoint_identity}
    spawn = mp.get_context("spawn")
    checkpoint_lock = spawn.Lock()
    # Persist a valid segment before either model is loaded.  This makes a
    # timeout during model download or CUDA initialization resumable too.
    _snapshot(
        checkpoint_spec=checkpoint_spec, checkpoint_lock=checkpoint_lock,
        states_dir=states_dir, reason="session-initialized",
    )
    worker_spec = {
        "base_model_path": worker.base_model_path,
        "tokenizer_path": str(worker.tokenizer_path),
        "adapter_path": str(worker.adapter_path),
        "config_raw": worker.config_raw,
        "config_path": str(getattr(worker.config, "path", "<e45-config>")),
        "config_sha256": worker.config_sha256,
    }
    processes: list[mp.Process] = []
    for rank, name in enumerate(("worker-0", "worker-1")):
        state = states[name]
        missing = [row for row in short[rank::2] if row["sample_index"] not in state["completed"]]
        if state["status"] == "oom":
            continue
        if not missing:
            continue
        process = spawn.Process(
            target=_replica_process,
            kwargs={
                "worker_spec": worker_spec, "contexts": short[rank::2], "device": f"cuda:{rank}",
                "state_path": str(states_dir / _REPLICA_STATE_NAMES[rank]),
                "checkpoint_spec": checkpoint_spec, "checkpoint_lock": checkpoint_lock,
                "deadline_epoch": deadline_epoch,
            },
        )
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
        if process.exitcode not in (0, None):
            raise PrivateGenerationResumeError(f"Private replica process exited with code {process.exitcode}")
    states = _write_initial_states(
        states_dir=states_dir, identity=identity, assignments=assignments,
        contexts_by_index=contexts_by_index,
    )
    # A state can be stopped in the second pass after it has already completed
    # all of its first-pass assignments.  That is a safe resume point, not a
    # reason to abandon a later session before it restarts those length rows.
    first_pass_stopped = [
        state for state in states.values()
        if state["status"] == "stopped" and set(state["completed"]) != set(state["assigned_indices"])
    ]
    if first_pass_stopped:
        return _checkpoint_outcome(
            checkpoint_spec=checkpoint_spec, checkpoint_lock=checkpoint_lock, states_dir=states_dir,
            states=states, total=len(contexts), reason="wall-clock-stop-during-replica-first-pass",
        )
    for name, state in states.items():
        if state["status"] == "stopped":
            state["status"] = "complete"
            state.pop("stop_reason", None)
            _persist_state(
                states_dir / (_SHARDED_STATE_NAME if name == "sharded-worker" else f"{name}-state.json"),
                state,
            )
    if any(state["status"] == "complete" for state in states.values()):
        _snapshot(
            checkpoint_spec=checkpoint_spec, checkpoint_lock=checkpoint_lock,
            states_dir=states_dir, reason="second-pass-resume-normalized",
        )
    fallback_rows: list[dict[str, Any]] = list(long)
    for rank, name in enumerate(("worker-0", "worker-1")):
        state = states[name]
        if state["status"] == "oom":
            fallback_rows.extend(
                row for row in short[rank::2] if row["sample_index"] not in state["completed"]
            )
        elif state["status"] != "complete":
            raise PrivateGenerationResumeError("Replica ended without complete, stopped, or approved OOM status")
    if fallback_rows:
        assignments["sharded-worker"] = [row["sample_index"] for row in fallback_rows]
        sharded_path = states_dir / _SHARDED_STATE_NAME
        if sharded_path.is_file():
            states = _refresh_states(
                states_dir=states_dir, identity=identity, assignments=assignments,
                contexts_by_index=contexts_by_index,
            )
            sharded = states["sharded-worker"]
        else:
            sharded = _make_state(
                worker_path="sharded", device=None, device_map="balanced", identity=identity,
                assigned_indices=assignments["sharded-worker"],
            )
            _persist_state(sharded_path, sharded)
            states["sharded-worker"] = sharded
            _snapshot(
                checkpoint_spec=checkpoint_spec, checkpoint_lock=checkpoint_lock,
                states_dir=states_dir, reason="sharded-fallback-initialized",
            )
        if sharded["status"] == "oom":
            raise PrivateGenerationResumeError("P01 sharded fallback OOM cannot be retried under this contract")
        if sharded["status"] != "complete":
            completed = _run_sharded_first_pass(
                worker=worker, state=sharded, state_path=sharded_path, rows=fallback_rows,
                long_indexes={row["sample_index"] for row in long}, checkpoint_spec=checkpoint_spec,
                checkpoint_lock=checkpoint_lock, deadline_epoch=deadline_epoch,
            )
            states = _refresh_states(
                states_dir=states_dir, identity=identity, assignments=assignments,
                contexts_by_index=contexts_by_index,
            )
            if not completed:
                return _checkpoint_outcome(
                    checkpoint_spec=checkpoint_spec, checkpoint_lock=checkpoint_lock, states_dir=states_dir,
                    states=states, total=len(contexts), reason="wall-clock-stop-during-sharded-first-pass",
                )
    states = _refresh_states(
        states_dir=states_dir, identity=identity, assignments=assignments,
        contexts_by_index=contexts_by_index,
    )
    provisional = _first_pass_results(states)
    if set(provisional) != set(range(len(contexts))):
        raise PrivateGenerationResumeError("First-pass workers did not produce exact private shard coverage")
    # First declare every length finish under its original worker state, then
    # restart only unfinished second-pass rows from the original prompt IDs.
    second_due: list[int] = []
    for index, (owner, result) in sorted(provisional.items()):
        state = states[owner]
        if result.get("initial_finish_reason") == "length":
            if index not in state["second_pass_completed"]:
                raise PrivateGenerationResumeError("Second-pass record has inconsistent completion state")
            continue
        if result["finish_reason"] == "length":
            if index not in state["second_pass_indices"]:
                state["second_pass_indices"].append(index)
            if index not in state["second_pass_completed"]:
                second_due.append(index)
    if second_due:
        for name, state in states.items():
            state["second_pass_indices"].sort()
            if state["second_pass_indices"]:
                state["second_pass_device_map"] = "balanced"
            _persist_state(
                states_dir / (_SHARDED_STATE_NAME if name == "sharded-worker" else f"{name}-state.json"),
                state,
            )
        _snapshot(
            checkpoint_spec=checkpoint_spec, checkpoint_lock=checkpoint_lock,
            states_dir=states_dir, reason="second-pass-initialized",
        )
        model: Any | None = None
        try:
            tokenizer = worker._tokenizer()
            model = worker._load_model(None, sharded=True)
            for index in second_due:
                owner, result = provisional[index]
                owner_state = states[owner]
                owner_path = states_dir / (_SHARDED_STATE_NAME if owner == "sharded-worker" else f"{owner}-state.json")
                if _deadline_reached(deadline_epoch):
                    # Preserve a replica OOM classification: it determines
                    # that its still-missing first-pass rows remain assigned to
                    # the approved sharded fallback on the next session.
                    if owner_state["status"] != "oom":
                        owner_state["status"] = "stopped"
                    owner_state["stop_reason"] = "wall_clock_safety_margin"
                    _persist_state(owner_path, owner_state)
                    return _checkpoint_outcome(
                        checkpoint_spec=checkpoint_spec, checkpoint_lock=checkpoint_lock, states_dir=states_dir,
                        states=states, total=len(contexts), reason="wall-clock-stop-during-second-pass",
                    )
                prepared = contexts_by_index[index]
                answer, finish, count, input_hash = worker.generate_single(
                    model, tokenizer, prepared["prompt"], worker.inference_cfg["second_pass_max_new_tokens"]
                )
                if input_hash != prepared["prompt_input_ids_sha256"]:
                    raise PrivateGenerationResumeError("Second pass did not use original P01 prompt input IDs")
                result.update({
                    "raw_answer": answer,
                    "initial_finish_reason": "length",
                    "finish_reason": finish,
                    "second_pass_tokens": count,
                    "worker_path": result["worker_path"] + "_pass2",
                })
                owner_state["records"][str(index)] = result
                owner_state["second_pass_completed"].append(index)
                _persist_state(owner_path, owner_state)
                _snapshot(
                    checkpoint_spec=checkpoint_spec, checkpoint_lock=checkpoint_lock,
                    states_dir=states_dir, reason=f"second-pass:record-{index}",
                )
        finally:
            if model is not None:
                del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    states = _refresh_states(
        states_dir=states_dir, identity=identity, assignments=assignments,
        contexts_by_index=contexts_by_index,
    )
    provisional = _first_pass_results(states)
    expected_second = {
        index for index, (_, item) in provisional.items()
        if item.get("initial_finish_reason") == "length"
    }
    completed_second = {
        index for state in states.values() for index in state["second_pass_completed"]
    }
    if completed_second != expected_second:
        raise PrivateGenerationResumeError("Second-pass coverage is incomplete after private resume")
    raw_path, clean_path, report = _finalize_records(
        contexts=contexts, states=states, output_dir=output_dir, arm_name=arm_name, identity=identity,
    )
    checkpoint = _snapshot(
        checkpoint_spec=checkpoint_spec, checkpoint_lock=checkpoint_lock,
        states_dir=states_dir, reason="generation-complete",
    )
    return PrivateGenerationOutcome(
        completed=True, checkpoint=checkpoint, completed_first_pass=len(contexts), total=len(contexts),
        raw_path=raw_path, clean_path=clean_path, report=report, reason="complete",
    )
