"""Exact P01-style paired generation with isolated replicas and durable identity binding."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import multiprocessing as mp
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .corpus import file_sha256
from .final_private_p01 import unified_clean
from .e45_parent_training import E45Config

LOG = logging.getLogger("e45_paired_generation")


class GenerationError(RuntimeError):
    """Raised when an E45 P01 arm cannot complete under the frozen contract."""


def compute_json_sha256(data: Any) -> str:
    """Hash canonical JSON for persisted record identities."""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temp.replace(path)


def _is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).casefold()
    return "cuda out of memory" in text or "outofmemoryerror" in type(exc).__name__.casefold()


def _finish_reason(token_ids: Any, eos_token_id: int | list[int] | None, limit: int) -> str:
    eos_ids = set(eos_token_id if isinstance(eos_token_id, list) else [eos_token_id])
    if len(token_ids) and int(token_ids[-1]) in eos_ids:
        return "eos"
    return "length" if len(token_ids) >= limit else "other"


class E45GeneratorWorker:
    """Load one pinned adapter and generate a P01 arm."""

    def __init__(self, *, base_model_path: str, tokenizer_path: Path, adapter_path: Path, config: Any):
        self.base_model_path = str(base_model_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.adapter_path = Path(adapter_path)
        self.config = config
        self.config_raw = getattr(config, "raw", config)
        self.model_inventory = config.section("model_inventory")
        self.inference_cfg = config.section("inference")
        self.base_revision = self.model_inventory["base_revision"]
        self.adapter_sha256 = file_sha256(self.adapter_path / "adapter_model.safetensors")
        self.tokenizer_sha256 = file_sha256(self.tokenizer_path / "tokenizer.json")
        self.decoding_sha256 = compute_json_sha256(self.inference_cfg)
        self.cleanup_sha256 = compute_json_sha256(config.section("unified_clean"))
        self.config_sha256 = getattr(config, "sha", getattr(config, "sha256", ""))

    def verify_runtime(self) -> dict[str, str]:
        """Require the frozen E45 runtime before any model is loaded."""
        observed: dict[str, str] = {}
        for package, expected in self.config.section("runtime").items():
            if package == "torch":
                import torch
                actual = str(torch.__version__)
            else:
                actual = importlib.metadata.version(package)
            observed[package] = actual
            if actual != expected:
                raise GenerationError(
                    f"Frozen runtime mismatch for {package}: observed {actual!r}, expected {expected!r}"
                )
        return observed

    def identity(self) -> dict[str, Any]:
        body = {
            "experiment_id": getattr(self.config, "experiment_id", "E45"),
            "config_sha256": self.config_sha256,
            "base_model": self.model_inventory["base_model"],
            "base_revision": self.base_revision,
            "tokenizer_sha256": self.tokenizer_sha256,
            "adapter_sha256": self.adapter_sha256,
            "decoding_sha256": self.decoding_sha256,
            "cleanup_sha256": self.cleanup_sha256,
        }
        body["identity_sha256"] = compute_json_sha256(body)
        return body

    def _load_model(self, device: str | None, *, sharded: bool = False) -> Any:
        """Load the frozen base revision and PEFT adapter on explicit P01 devices."""
        import torch
        from peft import PeftModel
        from transformers import Qwen2ForCausalLM

        required = ("adapter_model.safetensors", "adapter_config.json")
        if any(not (self.adapter_path / name).is_file() for name in required):
            raise GenerationError("Adapter directory must contain adapter_model.safetensors and adapter_config.json")
        if sharded:
            # P01 uses Accelerate's balanced two-T4 dispatch; `auto` can legally
            # place all layers on a single device and is therefore not equivalent.
            device_map: dict[str, Any] | str | None = "balanced" if torch.cuda.is_available() else None
        else:
            if device is None:
                raise GenerationError("Replica worker requires an explicit CUDA device")
            device_map = {"": device} if torch.cuda.is_available() else None
        base = Qwen2ForCausalLM.from_pretrained(
            self.base_model_path,
            revision=self.base_revision,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device_map,
            **({"max_memory": {0: "14GiB", 1: "14GiB"}} if sharded and torch.cuda.is_available() else {}),
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        model = PeftModel.from_pretrained(base, str(self.adapter_path), is_trainable=False)
        model.eval()
        if sharded and torch.cuda.is_available():
            device_map_observed = dict(getattr(base, "hf_device_map", {}) or {})
            # Reuse the P01 normalizer: Accelerate may report a CUDA placement
            # as either ``cuda:0`` or integer ``0`` depending on its version.
            from .final_public_e40 import _devices

            devices = _devices(device_map_observed)
            if devices != {"cuda:0", "cuda:1"}:
                raise GenerationError(f"P01 sharded fallback did not use exactly both T4s: {device_map_observed}")
        elif device is not None and torch.cuda.is_available():
            if any(str(parameter.device) != device for parameter in model.parameters()):
                raise GenerationError(f"P01 replica is not wholly resident on its assigned device {device}")
        return model

    def _tokenizer(self) -> Any:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(self.tokenizer_path), revision=self.base_revision, fix_mistral_conversions=False, trust_remote_code=False
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @staticmethod
    def generate_single(model: Any, tokenizer: Any, prompt: str, limit: int) -> tuple[str, str, int, str]:
        """Use P01's input-tokenization and greedy generation arguments exactly."""
        import torch

        device = next(model.parameters()).device
        tensors = {
            key: value.to(device)
            for key, value in tokenizer(prompt, add_special_tokens=False, return_tensors="pt").items()
        }
        with torch.inference_mode():
            generated = model.generate(
                **tensors,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.0,
                no_repeat_ngram_size=0,
                max_new_tokens=limit,
                use_cache=True,
            )
        new_ids = generated[0, tensors["input_ids"].shape[1] :]
        answer = tokenizer.decode(new_ids.detach().cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        if not answer:
            raise GenerationError("Pinned generator returned an empty answer")
        input_hash = hashlib.sha256(
            json.dumps(tensors["input_ids"][0].detach().cpu().tolist(), separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        # P01 classifies completion against the loaded model's generation config,
        # not a separately supplied tokenizer default.
        eos_token_id = getattr(model.generation_config, "eos_token_id", tokenizer.eos_token_id)
        return answer, _finish_reason(new_ids, eos_token_id, limit), len(new_ids), input_hash

    def execute_p01_generation(
        self, *, contexts: list[dict[str, Any]], output_dir: Path, arm_name: str
    ) -> tuple[Path, Path, dict[str, Any]]:
        """Delegate to the process-isolated P01 scheduler."""
        return execute_p01_generation(self, contexts=contexts, output_dir=output_dir, arm_name=arm_name)


def _replica_process(*, worker_spec: dict[str, Any], contexts: list[dict[str, Any]], device: str, state_path: str) -> None:
    """Run one short partition in an independently spawned GPU process."""
    path = Path(state_path)
    state: dict[str, Any] = {
        "schema_version": "1.0", "worker_path": f"replica_{device.rsplit(':', 1)[-1]}", "device": device,
        "identity": worker_spec["identity"], "assigned_indices": [row["sample_index"] for row in contexts],
        "completed": [], "records": {}, "status": "running",
    }
    _atomic_json(path, state)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_device(int(device.rsplit(":", 1)[-1]))
        worker = E45GeneratorWorker(
            base_model_path=worker_spec["base_model_path"],
            tokenizer_path=Path(worker_spec["tokenizer_path"]),
            adapter_path=Path(worker_spec["adapter_path"]),
            config=E45Config(
                raw=worker_spec["config_raw"],
                path=Path(worker_spec["config_path"]),
                sha256=worker_spec["config_sha256"],
            ),
        )
        worker.verify_runtime()
        tokenizer = worker._tokenizer()
        model = worker._load_model(device)
        for prepared in contexts:
            answer, finish, count, input_hash = worker.generate_single(
                model, tokenizer, prepared["prompt"], worker.inference_cfg["initial_max_new_tokens"]
            )
            index = prepared["sample_index"]
            state["records"][str(index)] = {
                "raw_answer": answer, "finish_reason": finish, "first_pass_tokens": count,
                "prompt_input_ids_sha256": input_hash, "worker_path": state["worker_path"],
            }
            state["completed"].append(index)
            _atomic_json(path, state)
        state["status"] = "complete"
    except Exception as exc:
        if _is_cuda_oom(exc):
            state.update({"status": "oom", "fallback_reason": "cuda_out_of_memory"})
        else:
            state.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(limit=10)})
    finally:
        _atomic_json(path, state)


def _read_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise GenerationError(f"Replica worker did not write state: {path}")
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("status") not in {"complete", "oom"}:
        raise GenerationError(f"Replica worker failed: {state.get('error_type')}: {state.get('error')}")
    return state


def _context_identity(row: dict[str, Any]) -> dict[str, str]:
    required = ("record_sha256", "question_sha256", "evidence_sha256", "prompt_sha256", "prompt_input_ids_sha256")
    missing = [name for name in required if not row.get(name)]
    if missing:
        raise GenerationError(f"Prepared context is missing immutable identity fields: {missing}")
    return {
        "context_record_sha256": row["record_sha256"], "question_sha256": row["question_sha256"],
        "evidence_sha256": row["evidence_sha256"], "prompt_sha256": row["prompt_sha256"],
        "prompt_input_ids_sha256": row["prompt_input_ids_sha256"],
    }


def execute_p01_generation(worker: E45GeneratorWorker, *, contexts: list[dict[str, Any]], output_dir: Path, arm_name: str) -> tuple[Path, Path, dict[str, Any]]:
    """Run concurrent short replicas, controlled sharded fallback, and length-only restart."""
    import torch

    if not contexts:
        raise GenerationError("Cannot generate an empty arm")
    worker.verify_runtime()
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise GenerationError(f"Generation output must be fresh: {output_dir}")
    output_dir.mkdir(parents=True)
    states_dir = output_dir / "worker-states"
    states_dir.mkdir()
    for index, row in enumerate(contexts):
        if row.get("sample_index") != index:
            raise GenerationError("Prepared contexts must be in exact sample-index order")
        _context_identity(row)
    if torch.cuda.is_available() and torch.cuda.device_count() < 2:
        raise GenerationError("Frozen P01 generation requires two visible GPUs")

    identity = worker.identity()
    threshold = worker.inference_cfg["replica_max_input_tokens"]
    short, long = ([row for row in contexts if row["input_tokens"] <= threshold], [row for row in contexts if row["input_tokens"] > threshold])
    worker_spec = {
        "base_model_path": worker.base_model_path, "tokenizer_path": str(worker.tokenizer_path),
        "adapter_path": str(worker.adapter_path), "config_raw": worker.config_raw,
        "config_path": str(getattr(worker.config, "path", "<e45-config>")),
        "config_sha256": worker.config_sha256, "identity": identity,
    }
    partitions = [short[0::2], short[1::2]]
    processes: list[mp.Process] = []
    state_paths: list[Path] = []
    spawn = mp.get_context("spawn")
    for rank, assigned in enumerate(partitions):
        state_path = states_dir / f"worker-{rank}-state.json"
        process = spawn.Process(target=_replica_process, kwargs={"worker_spec": worker_spec, "contexts": assigned, "device": f"cuda:{rank}", "state_path": str(state_path)})
        process.start(); processes.append(process); state_paths.append(state_path)
    for process in processes:
        process.join()
        if process.exitcode not in (0, None):
            raise GenerationError(f"Replica process exited with code {process.exitcode}")

    provisional: dict[int, dict[str, Any]] = {}
    worker_states: dict[str, dict[str, Any]] = {}
    fallback_rows: list[dict[str, Any]] = list(long)
    for rank, state_path in enumerate(state_paths):
        state = _read_state(state_path); worker_states[f"worker-{rank}"] = state
        provisional.update({int(index): value for index, value in state["records"].items()})
        lookup = {row["sample_index"]: row for row in partitions[rank]}
        fallback_rows.extend(lookup[index] for index in state["assigned_indices"] if index not in provisional)

    if fallback_rows:
        tokenizer = worker._tokenizer(); model = worker._load_model(None, sharded=True)
        sharded = {"schema_version": "1.0", "worker_path": "sharded", "device_map": "balanced", "identity": identity, "assigned_indices": [row["sample_index"] for row in fallback_rows], "completed": [], "records": {}, "status": "running"}
        sharded_path = states_dir / "sharded-worker-state.json"; _atomic_json(sharded_path, sharded)
        try:
            long_indexes = {row["sample_index"] for row in long}
            for prepared in fallback_rows:
                answer, finish, count, input_hash = worker.generate_single(model, tokenizer, prepared["prompt"], worker.inference_cfg["initial_max_new_tokens"])
                item = {"raw_answer": answer, "finish_reason": finish, "first_pass_tokens": count, "prompt_input_ids_sha256": input_hash, "worker_path": "sharded", "fallback_reason": "long_prompt" if prepared["sample_index"] in long_indexes else "replica_cuda_out_of_memory"}
                provisional[prepared["sample_index"]] = item; sharded["records"][str(prepared["sample_index"])] = item; sharded["completed"].append(prepared["sample_index"]); _atomic_json(sharded_path, sharded)
            sharded["status"] = "complete"
        finally:
            _atomic_json(sharded_path, sharded); del model
        worker_states["sharded-worker"] = sharded

    if set(provisional) != set(range(len(contexts))):
        raise GenerationError("Generation workers did not produce exact coverage")
    regenerate = [row for row in contexts if provisional[row["sample_index"]]["finish_reason"] == "length"]
    if regenerate:
        tokenizer = worker._tokenizer(); model = worker._load_model(None, sharded=True)
        for prepared in regenerate:
            answer, finish, count, input_hash = worker.generate_single(model, tokenizer, prepared["prompt"], worker.inference_cfg["second_pass_max_new_tokens"])
            if input_hash != prepared["prompt_input_ids_sha256"]:
                raise GenerationError("Second pass did not use the original P01 prompt input IDs")
            prior_finish = provisional[prepared["sample_index"]]["finish_reason"]
            provisional[prepared["sample_index"]].update({"raw_answer": answer, "initial_finish_reason": prior_finish, "finish_reason": finish, "second_pass_tokens": count, "worker_path": provisional[prepared["sample_index"]]["worker_path"] + "_pass2"})
        del model
        # The durable worker state that owns the first-pass record also records
        # the bounded sharded restart.  This avoids an unbound second execution.
        for state in worker_states.values():
            state["second_pass_indices"] = [
                row["sample_index"] for row in regenerate if row["sample_index"] in state.get("completed", [])
            ]
            if state["second_pass_indices"]:
                state["second_pass_device_map"] = "balanced"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    raw_rows, clean_rows = [], []
    for index, prepared in enumerate(contexts):
        result = provisional[index]
        if result["prompt_input_ids_sha256"] != prepared["prompt_input_ids_sha256"]:
            raise GenerationError("Generation input IDs differ from prepared P01 input IDs")
        raw = {"sample_index": index, "question_id": prepared["question_id"], **_context_identity(prepared), **identity, "worker_path": result["worker_path"], "raw_answer": result["raw_answer"], "initial_finish_reason": result.get("initial_finish_reason", result["finish_reason"]), "finish_reason": result["finish_reason"], "first_pass_tokens": result["first_pass_tokens"], "second_pass_tokens": result.get("second_pass_tokens", 0), "fallback_reason": result.get("fallback_reason", "none")}
        raw["record_sha256"] = compute_json_sha256(raw); raw_rows.append(raw)
        answer, trim = unified_clean(raw["raw_answer"])
        clean = {"sample_index": index, "question_id": raw["question_id"], "raw_record_sha256": raw["record_sha256"], "cleanup_sha256": identity["cleanup_sha256"], "clean_answer": answer, "cleanup_report": trim}
        clean["record_sha256"] = compute_json_sha256(clean); clean_rows.append(clean)
    raw_path, clean_path = output_dir / "raw-records.jsonl", output_dir / "clean-records.jsonl"
    _atomic_jsonl(raw_path, raw_rows); _atomic_jsonl(clean_path, clean_rows)
    for name, state in worker_states.items():
        state["raw_record_sha256s"] = [raw_rows[index]["record_sha256"] for index in state.get("completed", [])]
        state["state_sha256"] = compute_json_sha256({key: value for key, value in state.items() if key != "state_sha256"})
        _atomic_json(states_dir / ("sharded-worker-state.json" if name == "sharded-worker" else f"{name}-state.json"), state)
    report = {"schema_version": "1.0", "experiment_id": identity["experiment_id"], "arm_name": arm_name, "identity": identity, "raw_records_sha256": file_sha256(raw_path), "clean_records_sha256": file_sha256(clean_path), "sample_size": len(raw_rows), "initial_length_finish_count": sum(row["initial_finish_reason"] == "length" for row in raw_rows), "final_length_finish_count": sum(row["finish_reason"] == "length" for row in raw_rows), "second_pass_count": len(regenerate), "worker_state_files": sorted(path.name for path in states_dir.iterdir()), "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    _atomic_json(output_dir / "generation-report.json", report)
    return raw_path, clean_path, report


def run_arm_generation(**_: Any) -> Any:
    """Reject obsolete single-worker callers; only the paired executor is valid."""
    raise GenerationError("Use E45GeneratorWorker.execute_p01_generation; single-worker generation is prohibited")
