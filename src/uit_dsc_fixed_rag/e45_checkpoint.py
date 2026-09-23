"""E45 cross-session checkpoint manifest, validation, packaging, safe resume, and output archive engine."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .corpus import file_sha256

LOG = logging.getLogger("e45_checkpoint")

MANDATORY_CHECKPOINT_COMPONENTS = [
    "adapter_model.safetensors",
    "adapter_config.json",
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
]

TOKENIZER_FILENAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)


class CheckpointError(RuntimeError):
    """Raised when checkpoint manifest validation or resume invariants fail."""


def tokenizer_file_hashes(directory: Path) -> dict[str, str]:
    """Hash the persisted tokenizer surface in deterministic filename order."""
    directory = Path(directory)
    return {
        name: file_sha256(directory / name)
        for name in TOKENIZER_FILENAMES
        if (directory / name).is_file()
    }


def copy_frozen_tokenizer_files(
    source_directory: Path,
    destination_directory: Path,
    expected_files: dict[str, str],
) -> dict[str, str]:
    """Copy the already-verified tokenizer bytes into a checkpoint unchanged.

    ``save_pretrained`` is intentionally forbidden here.  Transformers may
    normalize or enrich tokenizer JSON while serializing an in-memory object,
    which is semantically harmless but breaks the byte identity that binds an
    E45 checkpoint to its materialized training records.
    """
    source_directory = Path(source_directory)
    destination_directory = Path(destination_directory)
    observed_source = tokenizer_file_hashes(source_directory)
    if not observed_source:
        raise CheckpointError("Frozen tokenizer source contains no recognized tokenizer files")
    if observed_source != expected_files:
        raise CheckpointError(
            "Frozen tokenizer source files do not match the expected tokenizer identity"
        )

    destination_directory.mkdir(parents=True, exist_ok=True)
    for name in TOKENIZER_FILENAMES:
        destination = destination_directory / name
        if destination.exists():
            if not destination.is_file() or destination.is_symlink():
                raise CheckpointError(f"Unsafe tokenizer destination member: {name}")
            destination.unlink()

    for name, expected_sha256 in expected_files.items():
        source = source_directory / name
        destination = destination_directory / name
        shutil.copyfile(source, destination)
        observed_sha256 = file_sha256(destination)
        if observed_sha256 != expected_sha256:
            raise CheckpointError(
                f"Frozen tokenizer copy verification failed for {name}: "
                f"{observed_sha256} != {expected_sha256}"
            )

    observed_destination = tokenizer_file_hashes(destination_directory)
    if observed_destination != expected_files:
        raise CheckpointError("Checkpoint tokenizer copy does not exactly match the frozen file map")
    return observed_destination


def tokenizer_aggregate_sha256(files: dict[str, str]) -> str:
    """Return the immutable aggregate identity for a tokenizer file manifest."""
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def safe_extract_archive(archive_path: Path, destination: Path) -> list[str]:
    """Extract archive strictly preventing zip-slip, symlinks, absolute paths, and bombs."""
    if not archive_path.is_file():
        raise CheckpointError(f"Archive not found: {archive_path}")

    destination.mkdir(parents=True, exist_ok=True)
    extracted_members: list[str] = []
    seen_names: set[str] = set()

    MAX_TOTAL_EXTRACTED_BYTES = 10 * 1024 * 1024 * 1024  # 10 GB
    total_bytes = 0

    with zipfile.ZipFile(archive_path, "r") as zf:
        for info in zf.infolist():
            name = info.filename
            if not name or name.endswith("/"):
                continue

            # Path traversal & absolute path rejection
            norm_name = os.path.normpath(name)
            if (
                norm_name.startswith("..")
                or os.path.isabs(norm_name)
                or "\\..\\" in norm_name
                or "/../" in norm_name
            ):
                raise CheckpointError(f"Path traversal or absolute path detected in archive: {name}")

            # Case collision check
            lower_name = norm_name.lower()
            if lower_name in seen_names:
                raise CheckpointError(f"Case collision or duplicate member detected: {name}")
            seen_names.add(lower_name)

            # Check symlinks (UNIX mode attr)
            if (info.external_attr >> 16) & 0o120000 == 0o120000:
                raise CheckpointError(f"Symlink detected in archive member: {name}")

            total_bytes += info.file_size
            if total_bytes > MAX_TOTAL_EXTRACTED_BYTES:
                raise CheckpointError("Decompression bomb protection triggered: total bytes exceed 10GB")

            target_path = destination / norm_name
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted_members.append(Path(norm_name).as_posix())

    return extracted_members


def create_checkpoint_manifest(
    checkpoint_dir: Path,
    expected_identity: dict[str, Any],
) -> dict[str, Any]:
    """Generate and write atomic checkpoint-manifest.json inside saved checkpoint directory."""
    if not checkpoint_dir.is_dir():
        raise CheckpointError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    trainer_state_file = checkpoint_dir / "trainer_state.json"
    if not trainer_state_file.is_file():
        raise CheckpointError("Missing mandatory trainer_state.json")
    try:
        state = json.loads(trainer_state_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CheckpointError(f"Invalid trainer_state.json: {exc}") from exc
    global_step = state.get("global_step", 0)

    # Check mandatory files
    for comp in MANDATORY_CHECKPOINT_COMPONENTS:
        p = checkpoint_dir / comp
        if not p.is_file():
            raise CheckpointError(f"Checkpoint missing mandatory file: {comp}")

    world_size = expected_identity.get("world_size", 2)
    rng_states = {}
    for r in range(world_size):
        rf = checkpoint_dir / f"rng_state_{r}.pth"
        if not rf.is_file():
            raise CheckpointError(f"Checkpoint missing mandatory RNG state for rank {r}: {rf.name}")
        rng_states[rf.name] = file_sha256(rf)

    # Tokenizer files
    tokenizer_files = tokenizer_file_hashes(checkpoint_dir)
    if not tokenizer_files:
        raise CheckpointError("Checkpoint missing tokenizer files")
    expected_tokenizer_files = expected_identity.get("tokenizer_files")
    if expected_tokenizer_files is not None and tokenizer_files != expected_tokenizer_files:
        raise CheckpointError("Checkpoint tokenizer files do not match the frozen tokenizer file map")
    tokenizer_aggregate = tokenizer_aggregate_sha256(tokenizer_files)
    if tokenizer_aggregate != expected_identity["tokenizer_aggregate_sha256"]:
        raise CheckpointError("Checkpoint tokenizer files do not match the frozen tokenizer identity")

    manifest = {
        "schema_version": "1.0",
        "experiment_id": expected_identity["experiment_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_dir": checkpoint_dir.name,
        "global_step": global_step,
        "expected_total_steps": expected_identity["expected_total_steps"],
        "config_sha256": expected_identity["config_sha256"],
        "code_manifest_sha256": expected_identity["code_manifest_sha256"],
        "training_records_jsonl_sha256": expected_identity["training_records_jsonl_sha256"],
        "aggregate_records_sha256": expected_identity["aggregate_records_sha256"],
        "base_model": expected_identity["base_model"],
        "base_revision": expected_identity["base_revision"],
        "lora_rank": expected_identity["lora_rank"],
        "lora_alpha": expected_identity["lora_alpha"],
        "lora_dropout": expected_identity["lora_dropout"],
        "target_modules": expected_identity["target_modules"],
        "expected_trainable_parameters": expected_identity["expected_trainable_parameters"],
        "world_size": world_size,
        "per_device_train_batch": expected_identity.get("per_device_train_batch", 1),
        "gradient_accumulation": expected_identity.get("gradient_accumulation", 4),
        "seed": expected_identity.get("seed", 20260830),
        "maximum_total_sequence": expected_identity.get("maximum_total_sequence", 8192),
        "tokenizer_sha256": expected_identity["tokenizer_sha256"],
        "tokenizer_aggregate_sha256": tokenizer_aggregate,
        "runtime_versions": expected_identity["runtime_versions"],
        "adapter_model_sha256": file_sha256(checkpoint_dir / "adapter_model.safetensors"),
        "adapter_config_sha256": file_sha256(checkpoint_dir / "adapter_config.json"),
        "optimizer_sha256": file_sha256(checkpoint_dir / "optimizer.pt"),
        "scheduler_sha256": file_sha256(checkpoint_dir / "scheduler.pt"),
        "trainer_state_sha256": file_sha256(checkpoint_dir / "trainer_state.json"),
        "rng_states": rng_states,
        "tokenizer_files": tokenizer_files,
    }

    manifest_file = checkpoint_dir / "checkpoint-manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOG.info("Wrote atomic checkpoint manifest to %s", manifest_file)
    return manifest


def validate_checkpoint_resume(
    checkpoint_dir: Path,
    expected_identity: dict[str, Any],
) -> dict[str, Any]:
    """Strictly validate that a checkpoint exactly matches the current experiment identity and all files."""
    if not checkpoint_dir.is_dir():
        raise CheckpointError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    manifest_file = checkpoint_dir / "checkpoint-manifest.json"
    if not manifest_file.is_file():
        raise CheckpointError(f"Missing checkpoint-manifest.json in {checkpoint_dir}")

    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CheckpointError(f"Corrupted checkpoint-manifest.json: {exc}") from exc

    required_matches = [
        "experiment_id",
        "config_sha256",
        "code_manifest_sha256",
        "training_records_jsonl_sha256",
        "aggregate_records_sha256",
        "base_model",
        "base_revision",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "target_modules",
        "expected_trainable_parameters",
        "world_size",
        "per_device_train_batch",
        "gradient_accumulation",
        "seed",
        "maximum_total_sequence",
        "expected_total_steps",
        "tokenizer_sha256",
        "tokenizer_aggregate_sha256",
        "runtime_versions",
    ]

    for field in required_matches:
        if field not in expected_identity:
            raise CheckpointError(f"expected_identity missing required field: {field}")
        if manifest.get(field) != expected_identity.get(field):
            raise CheckpointError(
                f"Checkpoint resume mismatch on {field}: manifest {manifest.get(field)} != expected {expected_identity.get(field)}"
            )

    global_step = manifest.get("global_step", 0)
    expected_total = manifest.get("expected_total_steps", 705)
    if not (0 < global_step <= expected_total):
        raise CheckpointError(f"Invalid global_step {global_step} (expected 1..{expected_total})")

    # Assert physical files match manifest hashes
    for comp in MANDATORY_CHECKPOINT_COMPONENTS:
        p = checkpoint_dir / comp
        if not p.is_file():
            raise CheckpointError(f"Checkpoint missing mandatory file: {comp}")
        h = file_sha256(p)
        key = f"{p.stem}_{p.suffix[1:]}_sha256" if "_" not in p.stem else f"{p.stem}_sha256"
        if p.name == "adapter_model.safetensors":
            key = "adapter_model_sha256"
        elif p.name == "adapter_config.json":
            key = "adapter_config_sha256"
        elif p.name == "optimizer.pt":
            key = "optimizer_sha256"
        elif p.name == "scheduler.pt":
            key = "scheduler_sha256"
        elif p.name == "trainer_state.json":
            key = "trainer_state_sha256"
        if manifest.get(key) != h:
            raise CheckpointError(f"Checkpoint hash mismatch on {comp}: observed {h} != manifest {manifest.get(key)}")

    trainer_state = json.loads((checkpoint_dir / "trainer_state.json").read_text(encoding="utf-8"))
    if trainer_state.get("global_step") != global_step:
        raise CheckpointError("Step mismatch between trainer_state.json and checkpoint-manifest.json")

    # Check RNG states
    world_size = manifest["world_size"]
    for r in range(world_size):
        rf = checkpoint_dir / f"rng_state_{r}.pth"
        if not rf.is_file():
            raise CheckpointError(f"Missing RNG state file: {rf.name}")
        h = file_sha256(rf)
        if manifest.get("rng_states", {}).get(rf.name) != h:
            raise CheckpointError(f"RNG state hash altered for rank {r}: {rf.name}")

    observed_tokenizer_files = tokenizer_file_hashes(checkpoint_dir)
    if observed_tokenizer_files != manifest.get("tokenizer_files"):
        raise CheckpointError("Checkpoint tokenizer file manifest changed")
    if tokenizer_aggregate_sha256(observed_tokenizer_files) != manifest.get("tokenizer_aggregate_sha256"):
        raise CheckpointError("Checkpoint tokenizer aggregate identity changed")

    allowed_files = set(MANDATORY_CHECKPOINT_COMPONENTS) | {
        "checkpoint-manifest.json",
        "training_args.bin",
    }
    for r in range(world_size):
        allowed_files.add(f"rng_state_{r}.pth")
    for name in manifest.get("tokenizer_files", {}):
        allowed_files.add(name)

    for item in checkpoint_dir.iterdir():
        if item.is_file() and item.name not in allowed_files:
            raise CheckpointError(f"Unexpected extra file found in checkpoint directory: {item.name}")

    return manifest


def package_checkpoint_bin(
    checkpoint_dir: Path,
    output_bin_path: Path,
    segment_report: dict[str, Any] | None = None,
) -> tuple[Path, str]:
    """Package checkpoint directory into safe .bin archive with .sha256 sidecar and segment report."""
    if not checkpoint_dir.is_dir():
        raise CheckpointError(f"Cannot package non-existent directory: {checkpoint_dir}")

    output_bin_path.parent.mkdir(parents=True, exist_ok=True)
    temp_bin = output_bin_path.with_suffix(".tmp")

    with zipfile.ZipFile(temp_bin, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(checkpoint_dir.rglob("*")):
            if file_path.is_file():
                rel_name = file_path.relative_to(checkpoint_dir).as_posix()
                zf.write(file_path, arcname=f"checkpoint/{rel_name}")

    temp_bin.replace(output_bin_path)
    bin_sha = file_sha256(output_bin_path)

    sidecar_path = Path(str(output_bin_path) + ".sha256")
    sidecar_path.write_text(f"{bin_sha}  {output_bin_path.name}\n", encoding="utf-8")

    if segment_report is not None:
        report_path = output_bin_path.parent / "E45_ACCOUNT_A_SEGMENT_REPORT.json"
        report_payload = {
            **segment_report,
            "checkpoint_bin": output_bin_path.name,
            "checkpoint_bin_sha256": bin_sha,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return output_bin_path, bin_sha


def build_account_output_archive(
    source_dir: Path,
    output_bin_path: Path,
    arm_type: str,  # 'candidate' or 'control'
) -> tuple[Path, str]:
    """Create strict, safe output archive (.bin) for Account A candidate or Account B control."""
    if not source_dir.is_dir():
        raise CheckpointError(f"Source directory does not exist: {source_dir}")

    output_bin_path.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, Any]] = {}

    allowed_top_level = {
        "holdout-contexts.jsonl",
        "raw-records.jsonl",
        "clean-records.jsonl",
        "generation-report.json",
        "runtime-identity.json",
    }
    if arm_type == "candidate":
        allowed_top_level.update({"complete.json", "preparation-manifest.json", "smoke_result.json"})

    required_top_level = {
        "holdout-contexts.jsonl", "raw-records.jsonl", "clean-records.jsonl",
        "generation-report.json", "runtime-identity.json",
    }
    if arm_type == "candidate":
        required_top_level.update({"complete.json", "preparation-manifest.json"})

    for item in source_dir.iterdir():
        if item.is_file():
            if item.name not in allowed_top_level and item.name != "manifest.json":
                raise CheckpointError(f"Unexpected top-level file in {arm_type} output: {item.name}")
        elif item.is_dir():
            if item.name == "adapter":
                if arm_type != "candidate":
                    raise CheckpointError("Account B control archive must not package adapter directory")
            elif item.name != "worker-states":
                raise CheckpointError(f"Unexpected directory in {arm_type} output: {item.name}")

    present_top_level = {item.name for item in source_dir.iterdir()}
    if not required_top_level.issubset(present_top_level) or "worker-states" not in present_top_level:
        raise CheckpointError(f"{arm_type} output is missing required identity/state artifacts")

    # Build manifest of all files
    for p in sorted(source_dir.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            if p.is_symlink():
                raise CheckpointError(f"Refusing symlink in output staging: {p}")
            rel = p.relative_to(source_dir).as_posix()
            manifest[rel] = {
                "bytes": p.stat().st_size,
                "sha256": file_sha256(p),
            }

    (source_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["manifest.json"] = {
        "bytes": (source_dir / "manifest.json").stat().st_size,
        "sha256": file_sha256(source_dir / "manifest.json"),
    }

    temp_bin = output_bin_path.with_suffix(".tmp")
    with zipfile.ZipFile(temp_bin, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel_path in sorted(manifest.keys()):
            zf.write(source_dir / rel_path, arcname=rel_path)

    temp_bin.replace(output_bin_path)
    bin_sha = file_sha256(output_bin_path)

    sidecar = Path(str(output_bin_path) + ".sha256")
    sidecar.write_text(f"{bin_sha}  {output_bin_path.name}\n", encoding="utf-8")
    LOG.info("Built %s output archive %s (SHA-256: %s)", arm_type, output_bin_path, bin_sha)
    return output_bin_path, bin_sha
