"""Durable, content-bound checkpoint archives for E45 private shards.

The archive intentionally contains only generated worker state and immutable
identity.  Prepared contexts are regenerated through the frozen answer-blind
P00 path on resume, then their hash must exactly match the checkpoint.  This
keeps every per-answer snapshot small enough to write during a Kaggle session
without weakening provenance.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from uit_dsc_fixed_rag.corpus import file_sha256
from uit_dsc_fixed_rag.e45_checkpoint import CheckpointError, safe_extract_archive


PRIVATE_CHECKPOINT_SCHEMA = "E45_PRIVATE_SHARD_CHECKPOINT_R4"
PRIVATE_CHECKPOINT_MANIFEST = "PRIVATE_CHECKPOINT_MANIFEST.json"
PRIVATE_FILE_MANIFEST = "FILE_MANIFEST.json"


class PrivateCheckpointError(RuntimeError):
    """Raised when a private checkpoint is missing, altered, or ambiguous."""


def canonical_sha256(value: Any) -> str:
    """Return the deterministic SHA-256 used by private checkpoint records."""
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(
        path,
        (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrivateCheckpointError(f"Invalid checkpoint JSON: {path.name}") from exc
    if not isinstance(value, dict):
        raise PrivateCheckpointError(f"Expected checkpoint object: {path.name}")
    return value


def _state_members(state_dir: Path) -> dict[str, Path]:
    if not state_dir.is_dir():
        raise PrivateCheckpointError("Checkpoint source lacks worker-state directory")
    members = {
        path.name: path
        for path in state_dir.iterdir()
        if path.is_file() and path.suffix == ".json" and not path.is_symlink()
    }
    if not members:
        raise PrivateCheckpointError("Checkpoint source has no worker-state files")
    if len(members) != len({name.casefold() for name in members}):
        raise PrivateCheckpointError("Checkpoint worker-state names case-collide")
    if any(not name.endswith("-state.json") for name in members):
        raise PrivateCheckpointError("Checkpoint worker-state name is outside the strict allowlist")
    return dict(sorted(members.items()))


def _sidecar_matches(archive: Path, digest: str) -> list[Path]:
    candidates = [Path(str(archive) + ".sha256")]
    candidates.extend(path for path in archive.parent.glob("*.sha256") if path not in candidates)
    matches: list[Path] = []
    for sidecar in candidates:
        if not sidecar.is_file():
            continue
        try:
            fields = sidecar.read_text(encoding="utf-8").strip().split()
        except OSError as exc:
            raise PrivateCheckpointError(f"Cannot read checkpoint sidecar: {sidecar.name}") from exc
        if fields and fields[0].casefold() == digest.casefold():
            matches.append(sidecar)
    return matches


def verify_checkpoint_sidecar(archive: Path) -> str:
    """Verify archive bytes and require exactly one nearby matching sidecar."""
    archive = Path(archive)
    if not archive.is_file():
        raise PrivateCheckpointError("Private checkpoint archive is missing")
    digest = file_sha256(archive)
    matches = _sidecar_matches(archive, digest)
    if len(matches) != 1:
        raise PrivateCheckpointError(
            f"Expected exactly one matching checkpoint sidecar for {archive.name}; found {len(matches)}"
        )
    return digest


def _file_manifest(root: Path, *, excluded: set[str]) -> dict[str, dict[str, Any]]:
    return {
        path.relative_to(root).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    }


def _validate_file_manifest(root: Path) -> None:
    manifest_path = root / PRIVATE_FILE_MANIFEST
    manifest = _read_json(manifest_path)
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual != set(manifest) | {PRIVATE_FILE_MANIFEST}:
        raise PrivateCheckpointError("Checkpoint members differ from FILE_MANIFEST")
    for relative, details in manifest.items():
        if not isinstance(relative, str) or not isinstance(details, dict):
            raise PrivateCheckpointError("Invalid checkpoint file manifest schema")
        path = root / relative
        if (
            not path.is_file()
            or path.stat().st_size != details.get("bytes")
            or file_sha256(path) != details.get("sha256")
        ):
            raise PrivateCheckpointError(f"Checkpoint file manifest mismatch: {relative}")


def write_private_checkpoint(
    *,
    archive_path: Path,
    checkpoint_identity: dict[str, Any],
    state_dir: Path,
    reason: str,
) -> dict[str, Any]:
    """Atomically snapshot all durable worker state into a compact .bin archive.

    Callers serialize concurrent invocations with a process-shared lock.  Each
    worker state is itself atomically written before this function is called.
    """
    archive_path = Path(archive_path)
    if archive_path.suffix != ".bin":
        raise PrivateCheckpointError("Private checkpoint must use a .bin filename")
    if not isinstance(reason, str) or not reason:
        raise PrivateCheckpointError("Checkpoint reason must be a non-empty string")
    state_members = _state_members(Path(state_dir))
    identity_sha = canonical_sha256(checkpoint_identity)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="e45_private_checkpoint_", dir=archive_path.parent) as temporary:
        stage = Path(temporary) / "stage"
        states_destination = stage / "worker-states"
        states_destination.mkdir(parents=True)
        state_hashes: dict[str, str] = {}
        for name, source in state_members.items():
            destination = states_destination / name
            # Đọc sau atomic rename sẽ thấy một phiên bản hoàn chỉnh,
            # kể cả khi worker còn lại vẫn đang chạy.
            destination.write_bytes(source.read_bytes())
            state_hashes[name] = file_sha256(destination)
        manifest_body = {
            "schema": PRIVATE_CHECKPOINT_SCHEMA,
            "checkpoint_identity": checkpoint_identity,
            "checkpoint_identity_sha256": identity_sha,
            "worker_state_files": state_hashes,
            "reason": reason,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        manifest_body["checkpoint_manifest_sha256"] = canonical_sha256(manifest_body)
        _atomic_json(stage / PRIVATE_CHECKPOINT_MANIFEST, manifest_body)
        file_manifest = _file_manifest(stage, excluded={PRIVATE_FILE_MANIFEST})
        _atomic_json(stage / PRIVATE_FILE_MANIFEST, file_manifest)
        temporary_archive = archive_path.with_suffix(".tmp")
        with zipfile.ZipFile(temporary_archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for relative in sorted(
                set(file_manifest) | {PRIVATE_FILE_MANIFEST}
            ):
                archive.write(stage / relative, relative)
        temporary_archive.replace(archive_path)
    digest = file_sha256(archive_path)
    _atomic_bytes(
        Path(str(archive_path) + ".sha256"),
        f"{digest}  {archive_path.name}\n".encode("ascii"),
    )
    return {
        "checkpoint_archive": str(archive_path),
        "checkpoint_archive_sha256": digest,
        "checkpoint_identity_sha256": identity_sha,
        "worker_state_count": len(state_members),
        "reason": reason,
    }


def _validate_checkpoint_root(root: Path, expected_identity: dict[str, Any]) -> dict[str, Any]:
    _validate_file_manifest(root)
    manifest = _read_json(root / PRIVATE_CHECKPOINT_MANIFEST)
    required = {
        "schema",
        "checkpoint_identity",
        "checkpoint_identity_sha256",
        "worker_state_files",
        "reason",
        "created_at_utc",
        "checkpoint_manifest_sha256",
    }
    if set(manifest) != required or manifest.get("schema") != PRIVATE_CHECKPOINT_SCHEMA:
        raise PrivateCheckpointError("Private checkpoint manifest schema changed")
    body = {key: value for key, value in manifest.items() if key != "checkpoint_manifest_sha256"}
    if manifest["checkpoint_manifest_sha256"] != canonical_sha256(body):
        raise PrivateCheckpointError("Private checkpoint manifest hash is invalid")
    if (
        manifest.get("checkpoint_identity") != expected_identity
        or manifest.get("checkpoint_identity_sha256") != canonical_sha256(expected_identity)
    ):
        raise PrivateCheckpointError("Private checkpoint identity differs from this shard execution")
    states_dir = root / "worker-states"
    state_members = _state_members(states_dir)
    hashes = {name: file_sha256(path) for name, path in state_members.items()}
    if hashes != manifest.get("worker_state_files"):
        raise PrivateCheckpointError("Private checkpoint worker-state hash mismatch")
    return manifest


def restore_private_checkpoint(
    *,
    archive_path: Path,
    destination_state_dir: Path,
    expected_identity: dict[str, Any],
) -> dict[str, Any]:
    """Safely extract one checkpoint into a fresh state directory and validate it."""
    archive_path = Path(archive_path)
    verify_checkpoint_sidecar(archive_path)
    destination_state_dir = Path(destination_state_dir)
    if destination_state_dir.exists() and any(destination_state_dir.iterdir()):
        raise PrivateCheckpointError("Private resume state destination must be empty")
    destination_state_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="e45_private_restore_", dir=destination_state_dir.parent) as temporary:
        extracted = Path(temporary) / "checkpoint"
        try:
            safe_extract_archive(archive_path, extracted)
        except CheckpointError as exc:
            raise PrivateCheckpointError("Private checkpoint archive failed safe extraction") from exc
        manifest = _validate_checkpoint_root(extracted, expected_identity)
        shutil.copytree(extracted / "worker-states", destination_state_dir)
    return manifest


def find_private_resume_checkpoint(
    *,
    search_roots: Iterable[Path],
    shard_index: int,
    shard_count: int,
) -> Path | None:
    """Find zero or exactly one checkpoint archive for the requested shard.

    Identification uses the archive member manifest, never its user-controlled
    basename.  This also tolerates Kaggle/browser renaming ZIP bytes to `.zip`.
    """
    candidates: list[Path] = []
    for raw_root in search_roots:
        root = Path(raw_root)
        if not root.exists():
            continue
        for candidate in root.rglob("*"):
            if not candidate.is_file() or candidate.suffix.casefold() not in {".bin", ".zip"}:
                continue
            try:
                with zipfile.ZipFile(candidate) as archive:
                    names = set(archive.namelist())
                    if PRIVATE_CHECKPOINT_MANIFEST not in names:
                        # Bundle đã hoàn tất không phải checkpoint để resume.
                        # Nếu gắn nhầm, dừng trước khi session mới sinh lại shard từ đầu.
                        if "shard-report.json" in names:
                            try:
                                report = json.loads(archive.read("shard-report.json").decode("utf-8"))
                            except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                                raise PrivateCheckpointError("Attached private final shard bundle is invalid") from exc
                            shard = report.get("shard_identity") if isinstance(report, dict) else None
                            if (
                                isinstance(shard, dict)
                                and shard.get("shard_index") == shard_index
                                and shard.get("shard_count") == shard_count
                            ):
                                raise PrivateCheckpointError(
                                    "A completed private shard bundle cannot be used as a resume checkpoint"
                                )
                        continue
                    raw_manifest = archive.read(PRIVATE_CHECKPOINT_MANIFEST)
            except (OSError, zipfile.BadZipFile, KeyError):
                continue
            try:
                value = json.loads(raw_manifest.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise PrivateCheckpointError("Attached private checkpoint has an invalid manifest") from exc
            identity = value.get("checkpoint_identity") if isinstance(value, dict) else None
            if not isinstance(identity, dict):
                continue
            if identity.get("shard_index") != shard_index or identity.get("shard_count") != shard_count:
                raise PrivateCheckpointError(
                    "Attached private checkpoint belongs to a different global-index shard"
                )
            candidates.append(candidate)
    if len(candidates) > 1:
        raise PrivateCheckpointError(
            "Attach zero or exactly one prior checkpoint for this private shard; "
            f"found {len(candidates)}"
        )
    if not candidates:
        return None
    verify_checkpoint_sidecar(candidates[0])
    return candidates[0]
