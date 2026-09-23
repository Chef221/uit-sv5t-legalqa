"""Static pre-Kaggle audit for the four generated resumable notebooks."""

from __future__ import annotations

import ast
import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SYSTEM = ROOT / "E45_PRIVATE_SHARDS_SYSTEM_R4_RESUMABLE.bin"
NOTEBOOKS = ROOT / "notebooks-resumable"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    assert SYSTEM.is_file()
    sidecar = Path(str(SYSTEM) + ".sha256")
    assert sidecar.read_text(encoding="ascii").split()[0] == sha256(SYSTEM)
    with zipfile.ZipFile(SYSTEM) as archive:
        names = archive.namelist()
        manifest = json.loads(archive.read("SYSTEM_FILE_MANIFEST.json"))
        assert len(names) == len(set(names))
        assert set(names) == set(manifest) | {"SYSTEM_FILE_MANIFEST.json"}
        assert all(
            not name.startswith(("/", "\\"))
            and ".." not in Path(name).parts
            and not name.endswith("/")
            for name in manifest
        )
        assert all(
            len(archive.read(name)) == details["bytes"]
            and hashlib.sha256(archive.read(name)).hexdigest() == details["sha256"]
            for name, details in manifest.items()
        )
        required = {
            "src/e45_private_checkpoint.py",
            "src/e45_private_resumable_generation.py",
            "src/e45_private_shards.py",
            "scripts/run_e45_private_shard.py",
        }
        assert required <= set(manifest)
        scheduler = archive.read("src/e45_private_resumable_generation.py").decode("utf-8")
        checkpoint = archive.read("src/e45_private_checkpoint.py").decode("utf-8")
        contexts = archive.read("src/uit_dsc_fixed_rag/e45_holdout_contexts.py").decode("utf-8")
        p01 = archive.read("src/uit_dsc_fixed_rag/e45_paired_generation.py").decode("utf-8")
        assert "worker.generate_single(" in scheduler
        assert "_load_model(None, sharded=True)" in scheduler
        assert "spawn.Process(" in scheduler
        assert "write_private_checkpoint(" in scheduler
        assert "add_special_tokens=False" in p01
        assert 'device_map = {"": device}' in p01
        assert 'device_map: dict[str, Any] | str | None = "balanced"' in p01
        assert "revision=self.base_revision" in p01
        assert "hit.chunk_id" in contexts
        assert 'h["chunk_id"]' not in contexts
        assert "Loading and validating pinned FAISS index" in contexts
        assert "safe_extract_archive(" in checkpoint
        assert "checkpoint_identity" in checkpoint

    notebooks = sorted(NOTEBOOKS.glob("E45-PRIVATE-RESUMABLE-SHARD-*-OF-4-T4X2.ipynb"))
    assert len(notebooks) == 4
    expected_sha = sha256(SYSTEM)
    for index, path in enumerate(notebooks):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        document = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        ast.parse(code)
        assert f"SHARD_INDEX, SHARD_COUNT = {index}, 4" in code
        assert expected_sha in code
        for required_text in (
            "import hashlib, json, os, shutil, stat, sys, zipfile",
            '"faiss-cpu": "1.15.0"',
            '"torchao"',
            "find_private_resume_checkpoint",
            "--checkpoint-archive",
            "--resume-checkpoint",
            "WALL_CLOCK_SAFETY_SECONDS = 39_000",
            "private_heartbeat_seconds",
            "input_hash_progress",
            "system_archive_hash_progress",
            "Save Version",
        ):
            assert required_text in (document if required_text == "Save Version" else code), (path.name, required_text)
        for forbidden in ("extractall(", "input_files[", "device_map=\"auto\"", "h[\"chunk_id\"]"):
            assert forbidden not in code, (path.name, forbidden)
    print(json.dumps({"status": "PASS", "system_sha256": expected_sha, "notebook_count": len(notebooks)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
