"""Cryptographic SHA-256 input resolver and safe system archive extractor for E45."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Iterable

LOG = logging.getLogger("e45_input_resolver")


class InputResolverError(Exception):
    """Raised when required inputs cannot be unambiguously resolved by content hash."""


E38_ADAPTER_MODEL_SHA256 = "e8f55f088fe2c5951c336095a5682af3a53f62c8be186bd5bfbf34741d895222"
E38_ADAPTER_CONFIG_SHA256 = "304011debb0daae761ed0f98da8c4c39f93510943a84af666a02b734595c1923"


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file in binary chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


class Sha256Index:
    """Recursively indexes files by SHA-256 content digest."""

    def __init__(self, search_roots: Iterable[Path]):
        self.sha_to_paths: dict[str, list[Path]] = {}
        self.indexed_files_count = 0
        self._build_index(search_roots)

    def _build_index(self, roots: Iterable[Path]) -> None:
        for root in roots:
            root_path = Path(root).resolve()
            if not root_path.exists():
                continue
            if root_path.is_file():
                self._add_file(root_path)
                continue
            for entry in root_path.rglob("*"):
                if entry.is_file():
                    # Skip symlinks and common temporary junk
                    if entry.is_symlink() or entry.name.startswith("."):
                        continue
                    self._add_file(entry)

    def _add_file(self, file_path: Path) -> None:
        try:
            sha = compute_file_sha256(file_path)
            self.sha_to_paths.setdefault(sha, []).append(file_path)
            self.indexed_files_count += 1
        except (OSError, PermissionError) as exc:
            LOG.warning("Could not read file for hashing: %s (%s)", file_path, exc)

    def find_matches(self, expected_sha256: str) -> list[Path]:
        return self.sha_to_paths.get(expected_sha256.lower(), [])

    def resolve_single(self, artifact_name: str, expected_sha256: str) -> Path:
        matches = self.find_matches(expected_sha256)
        if not matches:
            raise InputResolverError(
                f"Missing required artifact '{artifact_name}': "
                f"SHA-256 {expected_sha256} was not found in any indexed location."
            )
        if len(matches) > 1:
            paths_str = ", ".join(str(p) for p in matches)
            raise InputResolverError(
                f"Ambiguous artifact match for '{artifact_name}': "
                f"SHA-256 {expected_sha256} matches multiple files: {paths_str}"
            )
        resolved = matches[0]
        LOG.info("Resolved %s -> %s (SHA-256: %s)", artifact_name, resolved, expected_sha256)
        return resolved


def resolve_required_artifacts(
    search_roots: Iterable[Path],
    required_specs: dict[str, str],
) -> dict[str, Path]:
    """Resolve a dictionary of {artifact_name: expected_sha256} from search roots.

    Fails closed if any artifact is missing or matched multiple times.
    """
    index = Sha256Index(search_roots)
    LOG.info("Sha256Index indexed %d files across search roots", index.indexed_files_count)
    resolved: dict[str, Path] = {}
    for name, sha in required_specs.items():
        resolved[name] = index.resolve_single(name, sha)
    return resolved


def resolve_system_archive(
    search_roots: Iterable[Path],
    expected_archive_sha256: str,
) -> tuple[Path, Path]:
    """Find the system archive .bin and verify its .sha256 sidecar.

    Returns (archive_path, sidecar_path).
    """
    index = Sha256Index(search_roots)
    matches = index.find_matches(expected_archive_sha256)
    if not matches:
        raise InputResolverError(
            f"System archive with SHA-256 {expected_archive_sha256} not found in search roots."
        )
    if len(matches) > 1:
        raise InputResolverError(
            f"Multiple files match system archive SHA-256 {expected_archive_sha256}: "
            f"{[str(p) for p in matches]}"
        )
    archive_path = matches[0]

    # Look for sidecar next to archive or in index
    sidecar_candidate = archive_path.with_name(archive_path.name + ".sha256")
    if not sidecar_candidate.is_file():
        # Look for sidecar ending in .sha256 in same parent directory
        candidates = list(archive_path.parent.glob("*.sha256"))
        if candidates:
            sidecar_candidate = candidates[0]
        else:
            raise InputResolverError(f"Missing sidecar .sha256 for system archive {archive_path}")

    sidecar_content = sidecar_candidate.read_text(encoding="utf-8").strip()
    sidecar_hash = sidecar_content.split()[0].lower()
    if sidecar_hash != expected_archive_sha256.lower():
        raise InputResolverError(
            f"Sidecar hash {sidecar_hash} does not match expected system archive hash {expected_archive_sha256}"
        )

    LOG.info("System archive verified: %s (sidecar: %s)", archive_path, sidecar_candidate)
    return archive_path, sidecar_candidate


def extract_system_archive(archive_path: Path, target_dir: Path) -> Path:
    """Safely extract system archive into target directory, enforcing safe paths.

    Normalizes extraction so that `src/` and `configs/` exist directly under `target_dir`.
    """
    target_dir = Path(target_dir).resolve()
    if target_dir.exists() and any(target_dir.iterdir()):
        raise InputResolverError(f"Refusing to extract system archive into non-empty directory: {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path, "r") as zf:
        infolist = zf.infolist()
        # Check if all members share a common leading prefix (e.g. 'project/')
        names = [info.filename for info in infolist]
        prefix = ""
        if names and all(name.startswith("project/") for name in names):
            prefix = "project/"

        for info in infolist:
            name = info.filename
            if (
                name.startswith(("/", "\\"))
                or Path(name).is_absolute()
                or ".." in name.replace("\\", "/").split("/")
                or ((info.external_attr >> 16) & 0o170000) == 0o120000
            ):
                raise InputResolverError(f"Unsafe path in system archive: {name}")

            rel_name = name[len(prefix):] if prefix and name.startswith(prefix) else name
            if not rel_name or rel_name.endswith("/"):
                continue

            dest_path = (target_dir / rel_name).resolve()
            if not str(dest_path).startswith(str(target_dir)):
                raise InputResolverError(f"Path traversal detected in system archive: {name}")

            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, dest_path.open("wb") as dst:
                while chunk := src.read(1024 * 1024):
                    dst.write(chunk)

    # Validate essential layout
    required_paths = [
        target_dir / "src/uit_dsc_fixed_rag/__init__.py",
        target_dir / "configs/e45-inference-aligned-parent-lora-v1.json",
    ]
    for req in required_paths:
        if not req.is_file():
            raise InputResolverError(
                f"Extracted system archive layout invalid: missing {req} in {target_dir}"
            )

    LOG.info("Extracted system archive to %s with verified layout", target_dir)
    return target_dir


def build_verified_adapter_directory(
    *,
    search_roots: Iterable[Path],
    destination: Path,
    expected_model_sha256: str = E38_ADAPTER_MODEL_SHA256,
    expected_config_sha256: str = E38_ADAPTER_CONFIG_SHA256,
) -> Path:
    """Materialize an exact two-file PEFT adapter directory from hash-resolved inputs.

    PEFT resolves its adapter metadata relative to the weights directory.  Passing
    the arbitrary parent of a discovered weights file would silently couple the
    run to Kaggle's uploaded directory shape, so this function copies exactly
    the two pinned files into a fresh controlled directory.
    """
    resolved = resolve_required_artifacts(
        search_roots,
        {
            "frozen_e38_adapter_model": expected_model_sha256,
            "frozen_e38_adapter_config": expected_config_sha256,
        },
    )
    destination = Path(destination).resolve()
    if destination.exists():
        if any(destination.iterdir()):
            raise InputResolverError(f"Refusing to reuse non-empty adapter destination: {destination}")
    else:
        destination.mkdir(parents=True)

    model_dest = destination / "adapter_model.safetensors"
    config_dest = destination / "adapter_config.json"
    shutil.copy2(resolved["frozen_e38_adapter_model"], model_dest)
    shutil.copy2(resolved["frozen_e38_adapter_config"], config_dest)
    expected = {
        "adapter_model.safetensors": expected_model_sha256.lower(),
        "adapter_config.json": expected_config_sha256.lower(),
    }
    actual_names = {item.name for item in destination.iterdir() if item.is_file()}
    if actual_names != set(expected):
        raise InputResolverError(f"Controlled adapter directory has unexpected members: {sorted(actual_names)}")
    for name, digest in expected.items():
        if compute_file_sha256(destination / name) != digest:
            raise InputResolverError(f"Controlled adapter file hash mismatch: {name}")
    return destination


def build_verified_artifact_directory(
    *,
    search_roots: Iterable[Path],
    destination: Path,
    required_files: dict[str, str],
) -> Path:
    """Copy a hash-resolved artifact into a fresh directory with an exact allowlist.

    Kaggle assigns paths from user-controlled dataset slugs.  Runtime code must
    not infer an artifact root from any one resolved member, because files with
    the same basename can be uploaded in several datasets.  This helper makes a
    controlled root containing exactly the declared relative filenames.
    """
    if not required_files or any(Path(name).is_absolute() or ".." in Path(name).parts for name in required_files):
        raise InputResolverError("Artifact directory required file names must be safe relative paths")
    resolved = resolve_required_artifacts(search_roots, required_files)
    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise InputResolverError(f"Refusing to reuse non-empty artifact destination: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    for rel, source in resolved.items():
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    actual = {path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_file()}
    if actual != set(required_files):
        raise InputResolverError("Controlled artifact directory does not match its allowlist")
    for rel, expected in required_files.items():
        if compute_file_sha256(destination / rel) != expected.lower():
            raise InputResolverError(f"Controlled artifact file hash mismatch: {rel}")
    return destination


def resolve_and_extract_single_resume_checkpoint(
    *,
    search_roots: Iterable[Path],
    destination_parent: Path,
) -> Path | None:
    """Return one safely extracted checkpoint or ``None`` when no checkpoint is attached.

    A resume archive is optional, but accepting the first glob match is unsafe:
    a Kaggle input can contain multiple historical outputs.  This resolver
    requires every candidate to have a matching sidecar and allows at most one.
    The caller still validates the immutable run identity with
    :func:`validate_checkpoint_resume` before training.
    """
    candidates: list[Path] = []
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        candidates.extend(sorted(root.rglob("E45_ACCOUNT_A_CHECKPOINT_STEP_*.bin")))
    if not candidates:
        return None
    if len(candidates) != 1:
        raise InputResolverError(
            "Expected zero or one attached Account A checkpoint archive; "
            f"found {len(candidates)}: {[str(p) for p in candidates]}"
        )
    archive = candidates[0]
    sidecar = Path(str(archive) + ".sha256")
    if not sidecar.is_file():
        raise InputResolverError(f"Resume archive is missing sidecar: {sidecar}")
    declared = sidecar.read_text(encoding="utf-8").strip().split()[0].lower()
    actual = compute_file_sha256(archive)
    if declared != actual:
        raise InputResolverError(f"Resume archive sidecar mismatch: {actual} != {declared}")

    destination_parent = Path(destination_parent).resolve()
    destination_parent.mkdir(parents=True, exist_ok=True)
    destination = Path(tempfile.mkdtemp(prefix="e45_resume_", dir=destination_parent))
    try:
        from .e45_checkpoint import safe_extract_archive

        safe_extract_archive(archive, destination)
        checkpoint = destination / "checkpoint"
        if not checkpoint.is_dir() or any(p.name != "checkpoint" for p in destination.iterdir()):
            raise InputResolverError("Resume archive must contain exactly one top-level checkpoint/ directory")
        return checkpoint
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def compute_source_identity(project_root: Path) -> str:
    """Hash all shipped Python/config source bytes in deterministic path order."""
    project_root = Path(project_root)
    digest = hashlib.sha256()
    files = [
        path
        for relative in ("src", "configs", "scripts")
        for path in (project_root / relative).rglob("*")
        if path.is_file() and path.suffix in {".py", ".json", ".ps1"} and "__pycache__" not in path.parts
    ]
    for path in sorted(files, key=lambda item: item.relative_to(project_root).as_posix()):
        relative = path.relative_to(project_root).as_posix().encode("utf-8")
        digest.update(relative + b"\0" + compute_file_sha256(path).encode("ascii") + b"\n")
    return digest.hexdigest()


def build_runtime_identity(
    *,
    project_root: Path,
    config: Any,
    tokenizer_path: Path,
    contexts_path: Path,
    adapter_path: Path | None,
    arm_name: str,
) -> dict[str, Any]:
    """Build a complete immutable arm identity consumed by the local finalizer."""
    model = config.section("model_inventory")
    inference = config.section("inference")
    cleanup = config.section("unified_clean")
    versions = {}
    for name in (
        "torch", "transformers", "peft", "accelerate", "bitsandbytes",
        "sentence-transformers", "faiss-cpu", "numpy",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "MISSING"
    body = {
        "experiment_id": config.experiment_id,
        "arm_name": arm_name,
        "config_sha256": getattr(config, "sha", getattr(config, "sha256", "")),
        "code_sha256": compute_source_identity(project_root),
        "base_model": model["base_model"],
        "base_revision": model["base_revision"],
        "tokenizer_sha256": compute_file_sha256(Path(tokenizer_path) / "tokenizer.json"),
        "runtime_versions": versions,
        "decoding_sha256": hashlib.sha256(json.dumps(inference, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "cleanup_sha256": hashlib.sha256(json.dumps(cleanup, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "context_artifact_sha256": compute_file_sha256(contexts_path),
        "adapter_sha256": compute_file_sha256(Path(adapter_path) / "adapter_model.safetensors") if adapter_path else "none",
    }
    body["identity_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return body
