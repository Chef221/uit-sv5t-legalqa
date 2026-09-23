#!/usr/bin/env python3
"""Run E45 LoRA training with 2xT4 DDP, NF4 double quant, gradient checkpointing, and cross-session resume."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from uit_dsc_fixed_rag.corpus import file_sha256
from uit_dsc_fixed_rag.e45_checkpoint import (
    copy_frozen_tokenizer_files,
    create_checkpoint_manifest,
    package_checkpoint_bin,
    tokenizer_aggregate_sha256,
    tokenizer_file_hashes,
    validate_checkpoint_resume,
)
from uit_dsc_fixed_rag.e45_input_resolver import compute_source_identity
from uit_dsc_fixed_rag.e45_t4_attention import register_e45_t4_attention
from uit_dsc_fixed_rag.e45_parent_training import (
    E45CausalCollator,
    E45Error,
    E45TokenizedDataset,
    assert_model_inventory,
    load_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOG = logging.getLogger("train_kaggle")


def verify_frozen_runtime(config: Any) -> dict[str, str]:
    """Fail before model loading unless every runtime package is exact."""
    observed: dict[str, str] = {}
    for package, expected in config.section("runtime").items():
        actual = str(torch.__version__) if package == "torch" else importlib.metadata.version(package)
        observed[package] = actual
        if actual != expected:
            raise E45Error(f"Frozen runtime mismatch for {package}: {actual!r} != {expected!r}")
    return observed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", type=Path, default=PROJECT_ROOT / "configs/e45-inference-aligned-parent-lora-v1.json")
    parser.add_argument("--records-jsonl", type=Path, required=True, help="Path to materialized training-records.jsonl")
    parser.add_argument("--manifest-json", type=Path, required=True, help="Path to preparation-manifest.json")
    parser.add_argument("--base-model-path", type=str, required=True, help="Base model ID or local directory")
    parser.add_argument("--tokenizer-path", type=Path, required=True, help="Path to Vi-Qwen tokenizer")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store checkpoints and final adapter")
    parser.add_argument("--resume-checkpoint-dir", type=Path, default=None, help="Directory of existing checkpoint to resume")
    parser.add_argument("--max-runtime-hours", type=float, default=11.25, help="Safety margin wall-time in hours")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip memory smoke step (for testing only)")
    parser.add_argument("--allow-single-process-test", action="store_true", help="Allow running without DDP for unit tests only")
    return parser.parse_args()


class WallTimeGuardCallback(TrainerCallback):
    """Save atomic checkpoint and abort cleanly if session wall-time limit approaches."""

    def __init__(
        self,
        max_seconds: float,
        output_dir: Path,
        expected_identity: dict[str, Any],
        tokenizer_path: Path,
    ):
        self.start_time = time.time()
        self.max_seconds = max_seconds
        self.output_dir = output_dir
        self.expected_identity = expected_identity
        self.tokenizer_path = Path(tokenizer_path)
        self.aborted = False

    def on_step_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.start_time
        if elapsed >= self.max_seconds:
            LOG.warning("Wall-time safety threshold reached (%.2f / %.2f s). Stopping at step %d...", elapsed, self.max_seconds, state.global_step)
            control.should_save = True
            control.should_training_stop = True
            self.aborted = True

    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = self.output_dir / "checkpoints" / f"checkpoint-{state.global_step}"

        # Save per-rank RNG state
        rank = int(os.environ.get("RANK", "0"))
        rng_state = {
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        }
        torch.save(rng_state, ckpt_dir / f"rng_state_{rank}.pth")

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if state.is_world_process_zero and ckpt_dir.is_dir():
            LOG.info("Rank 0: copying frozen tokenizer bytes and creating verified checkpoint manifest for %s...", ckpt_dir)
            copy_frozen_tokenizer_files(
                self.tokenizer_path,
                ckpt_dir,
                self.expected_identity["tokenizer_files"],
            )
            manifest = create_checkpoint_manifest(ckpt_dir, self.expected_identity)
            bin_path = self.output_dir / f"E45_ACCOUNT_A_CHECKPOINT_STEP_{state.global_step}.bin"
            package_checkpoint_bin(ckpt_dir, bin_path, segment_report={
                "experiment_id": self.expected_identity["experiment_id"],
                "completed_optimizer_step": state.global_step,
                "expected_total_steps": self.expected_identity["expected_total_steps"],
                "manifest_sha256": file_sha256(ckpt_dir / "checkpoint-manifest.json"),
            })
            LOG.info("Packaged checkpoint archive %s", bin_path)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()


def run_max_length_smoke(
    dataset: E45TokenizedDataset,
    model: Any,
    collator: E45CausalCollator,
    gradient_accumulation_steps: int,
    expected_optimizer_steps: int,
    max_hours_threshold: float = 30.0,
    projection_safety_factor: float = 1.25,
) -> dict[str, Any]:
    """Smoke every non-dominated real batch by sequence and active-target size."""
    if gradient_accumulation_steps <= 0 or expected_optimizer_steps <= 0:
        raise E45Error("Smoke projection requires positive accumulation and optimizer-step counts")
    if projection_safety_factor < 1.0:
        raise E45Error("Smoke projection safety factor must be at least 1.0")

    # E45TokenizedDataset intentionally exposes only model input fields.  Use
    # the actual model input length rather than reaching into provenance fields
    # that __getitem__ does not return.
    row_shapes = [
        (
            len(dataset[i]["input_ids"]),
            sum(label != -100 for label in dataset[i]["labels"]),
            i,
        )
        for i in range(len(dataset))
    ]
    # Attention memory is monotone in total sequence length; LM-head/loss
    # memory is monotone in active answer+EOS labels.  Testing every Pareto
    # frontier shape covers the actual worst cases without 5,636 smoke steps.
    risk_cases = [
        candidate
        for candidate in row_shapes
        if not any(
            other[0] >= candidate[0]
            and other[1] >= candidate[1]
            and (other[0] > candidate[0] or other[1] > candidate[1])
            for other in row_shapes
        )
    ]
    risk_cases.sort()
    longest_tokens = max(shape[0] for shape in risk_cases)

    # from_pretrained() returns an evaluation-mode model.  Gradient
    # checkpointing only takes effect in training mode, so an eval-mode smoke
    # drastically overstates activation memory and can OOM even when the real
    # Trainer step fits.  The smoke must exercise the exact training path.
    model.train()
    device = next(model.parameters()).device if torch.cuda.is_available() else torch.device("cpu")
    # AdamW states appear only on the first optimizer step.  Allocate them on
    # this discarded stack before measuring the frontier; otherwise a smoke
    # can pass with a memory margin that disappears at real step one.
    smoke_optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-4,
        weight_decay=0.0,
    )
    smoke_cases: list[dict[str, Any]] = []
    execution_cases = [("optimizer-state-warmup", risk_cases[0])] + [
        ("frontier", case) for case in risk_cases
    ]
    for case_kind, (total_tokens, active_target_tokens, row_index) in execution_cases:
        batch = collator([dataset[row_index]])
        batch = {k: v.to(device) for k, v in batch.items()}
        LOG.info(
            "Running memory smoke frontier row %d/%d (%d total, %d active target tokens)...",
            len(smoke_cases) + 1,
            len(execution_cases),
            total_tokens,
            active_target_tokens,
        )
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        smoke_optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        peak = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0.0
        smoke_cases.append(
            {
                "row_index": row_index,
                "case_kind": case_kind,
                "total_tokens": total_tokens,
                "active_target_tokens": active_target_tokens,
                "step_time_seconds": elapsed,
                "peak_memory_mb": peak,
            }
        )
        model.zero_grad(set_to_none=True)
        del batch, outputs, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    measured_frontier = [case for case in smoke_cases if case["case_kind"] == "frontier"]
    peak_mb = max(case["peak_memory_mb"] for case in measured_frontier)

    # Estimate total runtime from equal-sized strata of the actual 5,636-row
    # shape distribution.  Multiplying the single slowest 8k frontier row by
    # every microbatch is a memory upper bound, not a runtime estimate.  The
    # deterministic proxy covers both quadratic attention work and active
    # vocabulary-logit work; a 25% factor below remains the declared margin.
    timing_sample_count = min(16, len(row_shapes))
    ranked_shapes = sorted(
        row_shapes,
        key=lambda shape: (shape[0] * shape[0] + shape[1] * 8192, shape[0], shape[1], shape[2]),
    )
    timing_shapes = [
        ranked_shapes[min(len(ranked_shapes) - 1, int((ordinal + 0.5) * len(ranked_shapes) / timing_sample_count))]
        for ordinal in range(timing_sample_count)
    ]
    timing_cases: list[dict[str, Any]] = []
    model.zero_grad(set_to_none=True)
    for ordinal, (total_tokens, active_target_tokens, row_index) in enumerate(timing_shapes):
        LOG.info(
            "Running stratified timing row %d/%d (%d total, %d active target tokens)...",
            ordinal + 1,
            len(timing_shapes),
            total_tokens,
            active_target_tokens,
        )
        batch = collator([dataset[row_index]])
        batch = {key: value.to(device) for key, value in batch.items()}
        perform_optimizer_step = (ordinal + 1) % gradient_accumulation_steps == 0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = model(**batch)
        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()
        if perform_optimizer_step:
            smoke_optimizer.step()
            model.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        timing_cases.append(
            {
                "row_index": row_index,
                "total_tokens": total_tokens,
                "active_target_tokens": active_target_tokens,
                "optimizer_step": perform_optimizer_step,
                "elapsed_seconds": elapsed,
            }
        )
        del batch, outputs, loss
    model.zero_grad(set_to_none=True)

    step_time = sum(case["elapsed_seconds"] for case in timing_cases) / len(timing_cases)
    projected_hours = (
        step_time
        * gradient_accumulation_steps
        * expected_optimizer_steps
        * projection_safety_factor
    ) / 3600.0

    LOG.info(
        "Smoke completed: stratified_mean_microbatch_time=%.3fs, peak_memory=%.1fMB, "
        "projected_total_hours=%.2f (limit: %.1fh)",
        step_time, peak_mb, projected_hours, max_hours_threshold,
    )
    if projected_hours > max_hours_threshold:
        raise E45Error(
            f"Projected training time ({projected_hours:.2f}h) exceeds maximum allowed threshold ({max_hours_threshold}h). "
            f"Training aborted fail-closed without altering scientific contract."
        )

    return {
        "longest_row_tokens": longest_tokens,
        "frontier_case_count": len(measured_frontier),
        "optimizer_state_warmup_count": 1,
        "smoke_cases": smoke_cases,
        "timing_sample_count": len(timing_cases),
        "timing_cases": timing_cases,
        "stratified_mean_microbatch_time_seconds": step_time,
        "step_time_seconds": step_time,
        "peak_memory_mb": peak_mb,
        "projected_hours": projected_hours,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "expected_optimizer_steps": expected_optimizer_steps,
        "projection_safety_factor": projection_safety_factor,
    }


def main() -> int:
    args = parse_args()
    LOG.info("Starting E45 LoRA training initialization...")
    config = load_config(args.config_path)
    training_cfg = config.section("training")
    model_inv = config.section("model_inventory")

    # 1. Enforce DDP World Size Contract
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not args.allow_single_process_test and world_size != training_cfg["world_size"]:
        raise E45Error(
            f"Live world size {world_size} != expected {training_cfg['world_size']}. "
            f"Account A must be launched with 'torchrun --nproc_per_node=2 scripts/run_e45_train_kaggle.py'."
        )

    # Effective global batch: 1 * 4 * 2 = 8
    global_batch = training_cfg["per_device_train_batch"] * training_cfg["gradient_accumulation"] * world_size
    planned_steps = (training_cfg["records_count"] + global_batch - 1) // global_batch
    if planned_steps != training_cfg["expected_optimizer_steps"]:
        raise E45Error(
            f"Planned optimizer steps {planned_steps} != expected {training_cfg['expected_optimizer_steps']} "
            f"(records={training_cfg['records_count']}, global_batch={global_batch})"
        )

    # Transformers 5.16 removed ``warmup_ratio`` from TrainingArguments and
    # accepts the same ratio as a float in ``warmup_steps``.  Construct and
    # validate the complete operator API before any model download or smoke so
    # runtime incompatibilities fail in seconds rather than after GPU work.
    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=training_cfg["epochs"],
        per_device_train_batch_size=training_cfg["per_device_train_batch"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation"],
        learning_rate=training_cfg["learning_rate"],
        warmup_steps=training_cfg["warmup_ratio"],
        lr_scheduler_type=training_cfg["scheduler"],
        weight_decay=training_cfg["weight_decay"],
        save_strategy="steps",
        save_steps=training_cfg["save_interval_steps"],
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        logging_steps=8,
        seed=training_cfg["seed"],
        data_seed=training_cfg["seed"],
        report_to="none",
    )
    expected_warmup_steps = math.ceil(
        training_cfg["expected_optimizer_steps"] * training_cfg["warmup_ratio"]
    )
    observed_warmup_steps = training_args.get_warmup_steps(training_cfg["expected_optimizer_steps"])
    if observed_warmup_steps != expected_warmup_steps:
        raise E45Error(
            f"TrainingArguments warmup resolution {observed_warmup_steps} != expected {expected_warmup_steps}"
        )

    # Check package identities only after the process topology has been
    # rejected or accepted.  This keeps the DDP-contract test independent of
    # whatever CPU-only runtime happens to execute the negative-path fixture.
    runtime_versions = verify_frozen_runtime(config)

    # 2. Verify static preparation manifest
    prep_manifest = json.loads(args.manifest_json.read_text(encoding="utf-8"))
    if not prep_manifest.get("all_gates_pass"):
        raise E45Error("Training aborted: static preparation gate did not pass.")
    if prep_manifest.get("config_sha256") != config.sha256:
        raise E45Error("Training aborted: preparation manifest config identity differs.")
    if prep_manifest.get("training_records_jsonl_sha256") != file_sha256(args.records_jsonl):
        raise E45Error("Training aborted: materialized training JSONL hash differs from preparation manifest.")

    records = [json.loads(line) for line in args.records_jsonl.open("r", encoding="utf-8")]
    if len(records) != training_cfg["records_count"]:
        raise E45Error(f"Records count {len(records)} != expected {training_cfg['records_count']}")

    dataset = E45TokenizedDataset(records)
    base_rev = model_inv["base_revision"]
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer_path),
        revision=base_rev,
        fix_mistral_conversions=False,
    )
    tokenizer_sha256 = file_sha256(args.tokenizer_path / "tokenizer.json")
    tokenizer_files = tokenizer_file_hashes(args.tokenizer_path)
    tokenizer_aggregate = tokenizer_aggregate_sha256(tokenizer_files)
    if prep_manifest.get("tokenizer_hash") != tokenizer_sha256 or prep_manifest.get("tokenizer_aggregate_sha256") != tokenizer_aggregate:
        raise E45Error("Training aborted: tokenizer identity differs from the preparation manifest.")
    collator = E45CausalCollator(tokenizer)

    # 3. Expected identity for checkpoints and resume
    expected_identity = {
        "experiment_id": config.experiment_id,
        "config_sha256": config.sha256,
        "code_manifest_sha256": compute_source_identity(PROJECT_ROOT),
        "training_records_jsonl_sha256": prep_manifest["training_records_jsonl_sha256"],
        "aggregate_records_sha256": prep_manifest["aggregate_records_sha256"],
        "base_model": model_inv["base_model"],
        "base_revision": base_rev,
        "lora_rank": training_cfg["lora_rank"],
        "lora_alpha": training_cfg["lora_alpha"],
        "lora_dropout": training_cfg["lora_dropout"],
        "target_modules": training_cfg["target_modules"],
        "expected_trainable_parameters": model_inv["expected_lora_trainable_parameters"],
        "world_size": world_size,
        "per_device_train_batch": training_cfg["per_device_train_batch"],
        "gradient_accumulation": training_cfg["gradient_accumulation"],
        "seed": training_cfg["seed"],
        "maximum_total_sequence": training_cfg["maximum_total_sequence"],
        "expected_total_steps": training_cfg["expected_optimizer_steps"],
        "tokenizer_sha256": tokenizer_sha256,
        "tokenizer_files": tokenizer_files,
        "tokenizer_aggregate_sha256": tokenizer_aggregate,
        "runtime_versions": runtime_versions,
    }

    if args.resume_checkpoint_dir is not None:
        LOG.info("Validating resume checkpoint: %s", args.resume_checkpoint_dir)
        validate_checkpoint_resume(args.resume_checkpoint_dir, expected_identity)
        LOG.info("Resume checkpoint validated fail-closed.")

    # 4. Quantization config: NF4 double quant, float16 compute
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    ) if torch.cuda.is_available() else None

    # 5. Build a fresh base+LoRA stack.  The smoke stack is explicitly discarded
    # before this factory is used for the real optimizer run.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    def fresh_lora_stack() -> Any:
        LOG.info("Loading fresh base model %s (revision: %s, local rank=%d)...", args.base_model_path, base_rev, local_rank)
        attention_implementation = register_e45_t4_attention()
        model_kwargs: dict[str, Any] = {
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            "trust_remote_code": False,
            "revision": base_rev,
            "attn_implementation": attention_implementation,
        }
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config
            # This is rank-local placement for DDP, never model-parallel auto dispatch.
            model_kwargs["device_map"] = {"": local_rank}
        base = AutoModelForCausalLM.from_pretrained(args.base_model_path, **model_kwargs)
        # KV cache is an inference optimization and is incompatible with the
        # gradient-checkpointed training path.  Disabling it changes no loss,
        # labels, optimizer step, or quality-affecting contract field.
        base.config.use_cache = False
        if torch.cuda.is_available():
            # PyTorch recommends the non-reentrant checkpoint implementation
            # for DDP.  PEFT also avoids the legacy input-requires-grad hook in
            # this mode, saving the small but decisive activation allocation at
            # 8,192 tokens while preserving the same forward, loss, RNG state,
            # optimizer recipe, and recomputed backward activations.
            base = prepare_model_for_kbit_training(
                base,
                use_gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        model = get_peft_model(base, lora_config)
        model.train()
        return model

    # 6. Apply fresh LoRA config
    lora_config = LoraConfig(
        r=training_cfg["lora_rank"],
        lora_alpha=training_cfg["lora_alpha"],
        target_modules=training_cfg["target_modules"],
        lora_dropout=training_cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    # 7. Memory smoke on a separately loaded real longest row.  It records a
    # conservative projection and cannot alter the model subsequently trained.
    if not args.skip_smoke and torch.cuda.is_available():
        smoke_model = fresh_lora_stack()
        smoke_inv = assert_model_inventory(smoke_model, config)
        smoke_result = run_max_length_smoke(
            dataset,
            smoke_model,
            collator,
            gradient_accumulation_steps=training_cfg["gradient_accumulation"],
            expected_optimizer_steps=training_cfg["expected_optimizer_steps"],
        )
        del smoke_model
        torch.cuda.empty_cache()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if int(os.environ.get("RANK", "0")) == 0:
            smoke_result["model_inventory"] = smoke_inv
            smoke_result["projection_decision"] = "PASS_BELOW_30_HOURS"
            (args.output_dir / "smoke_result.json").write_text(json.dumps(smoke_result, indent=2), encoding="utf-8")

    # 8. Fresh untouched stack for the actual DDP optimizer run.
    model = fresh_lora_stack()
    inv = assert_model_inventory(model, config)
    LOG.info("Model inventory assertion passed: %s", inv)

    # 9. Training arguments were constructed before model work as an exact
    # pinned-runtime API preflight.
    guard_callback = WallTimeGuardCallback(
        max_seconds=args.max_runtime_hours * 3600.0,
        output_dir=args.output_dir,
        expected_identity=expected_identity,
        tokenizer_path=args.tokenizer_path,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[guard_callback],
    )

    LOG.info("Starting training loop...")
    resume_from = str(args.resume_checkpoint_dir) if args.resume_checkpoint_dir else None
    train_result = trainer.train(resume_from_checkpoint=resume_from)

    # If completed without wall-time abort, save final adapter
    if not guard_callback.aborted:
        final_adapter_dir = args.output_dir / "adapter-final"
        final_adapter_dir.mkdir(parents=True, exist_ok=True)
        if trainer.is_world_process_zero():
            LOG.info("Saving final adapter to %s...", final_adapter_dir)
            model.save_pretrained(str(final_adapter_dir))
            copy_frozen_tokenizer_files(
                args.tokenizer_path,
                final_adapter_dir,
                tokenizer_files,
            )

            complete_payload = {
                "experiment_id": config.experiment_id,
                "config_sha256": config.sha256,
                "global_step": train_result.global_step,
                "expected_total_steps": training_cfg["expected_optimizer_steps"],
                "train_loss": train_result.training_loss,
                "adapter_model_sha256": file_sha256(final_adapter_dir / "adapter_model.safetensors"),
                "adapter_config_sha256": file_sha256(final_adapter_dir / "adapter_config.json"),
                "code_manifest_sha256": expected_identity["code_manifest_sha256"],
                "training_records_jsonl_sha256": expected_identity["training_records_jsonl_sha256"],
                "aggregate_records_sha256": expected_identity["aggregate_records_sha256"],
                "tokenizer_sha256": tokenizer_sha256,
                "tokenizer_aggregate_sha256": tokenizer_aggregate,
                "base_model": model_inv["base_model"],
                "base_revision": base_rev,
                "lora_rank": training_cfg["lora_rank"],
                "lora_alpha": training_cfg["lora_alpha"],
                "lora_dropout": training_cfg["lora_dropout"],
                "target_modules": training_cfg["target_modules"],
                "runtime_versions": runtime_versions,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            (args.output_dir / "complete.json").write_text(json.dumps(complete_payload, indent=2), encoding="utf-8")
            LOG.info("Training complete and certified.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
