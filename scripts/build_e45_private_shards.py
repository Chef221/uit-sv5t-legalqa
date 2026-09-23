#!/usr/bin/env python3
"""Build the isolated four-shard E45 private Kaggle package."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
E45_R3 = ROOT.parent / "implementation_r3"
SYSTEM = ROOT / "E45_PRIVATE_SHARDS_SYSTEM_R4_RESUMABLE.bin"
NOTEBOOKS = ROOT / "notebooks-resumable"
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _zip_info(name: str) -> zipfile.ZipInfo:
    """Return canonical metadata so identical inputs produce identical ZIP bytes."""
    info = zipfile.ZipInfo(name, date_time=ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (0o100644 & 0xFFFF) << 16
    info.flag_bits |= 0x800
    return info


def _cell(source: str, kind: str = "code") -> dict[str, object]:
    result: dict[str, object] = {"cell_type": kind, "metadata": {}, "source": source.strip("\n").splitlines(keepends=True)}
    if kind == "code":
        result.update({"execution_count": None, "outputs": []})
    return result


def build_system() -> str:
    """Inject private-shard code into a byte-verified copy of E45 R3 system code."""
    source = E45_R3 / "E45_TWO_ACCOUNT_SYSTEM_R3.bin"
    sidecar = Path(str(source) + ".sha256")
    if not source.is_file() or not sidecar.is_file() or sidecar.read_text(encoding="utf-8").split()[0].lower() != sha256(source):
        raise SystemExit("E45 R3 system archive or sidecar is missing or invalid.")
    members: dict[str, bytes] = {}
    with zipfile.ZipFile(source) as archive:
        manifest = json.loads(archive.read("SYSTEM_FILE_MANIFEST.json"))
        for name, details in manifest.items():
            data = archive.read(name)
            if len(data) != details["bytes"] or hashlib.sha256(data).hexdigest() != details["sha256"]:
                raise SystemExit(f"Base system member mismatch: {name}")
            members[name] = data
    for relative in (
        "src/e45_private_checkpoint.py",
        "src/e45_private_resumable_generation.py",
        "src/e45_private_shards.py",
        "scripts/run_e45_private_shard.py",
        "scripts/merge_e45_private_shards.py",
    ):
        source_path = ROOT / relative
        if not source_path.is_file():
            raise SystemExit(f"Missing private system source: {source_path}")
        members[relative] = source_path.read_bytes()
    manifest = {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} for name, data in sorted(members.items())}
    temporary = SYSTEM.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(members):
            archive.writestr(_zip_info(name), members[name], compresslevel=9)
        manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8")
        archive.writestr(_zip_info("SYSTEM_FILE_MANIFEST.json"), manifest_bytes, compresslevel=9)
    temporary.replace(SYSTEM)
    digest = sha256(SYSTEM)
    Path(str(SYSTEM) + ".sha256").write_text(f"{digest}  {SYSTEM.name}\n", encoding="ascii")
    return digest


def _notebook(shard: int, system_sha: str) -> dict[str, object]:
    markdown = f"""# E45 private candidate — resumable shard {shard} of 4

This notebook requires one local E45 release token issued only after Account A
has produced a complete 705-step candidate archive. It never reads private
answers. Select **GPU T4 x2**, enable Internet, then use **Save Version → Save & Run All → Always Save Output**.
"""
    environment = r'''
# Match the Account-A/B runtime preflight that has already passed on Kaggle.
# Do not replace torch: Kaggle must expose the frozen CUDA build.
import importlib.metadata, subprocess, sys
PIP_PINS = {
    "transformers": "5.16.1", "peft": "0.19.1", "accelerate": "1.13.0",
    "bitsandbytes": "0.50.2", "sentence-transformers": "5.4.1",
    "faiss-cpu": "1.15.0", "numpy": "2.0.2",
}
installed = {dist.metadata["Name"].lower(): dist.version for dist in importlib.metadata.distributions()}
need = [f"{name}=={version}" for name, version in PIP_PINS.items() if installed.get(name) != version]
if need:
    print({"installing_missing_or_wrong_pins": need}, flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", *need], check=True)

# transformers 5.16 rejects Kaggle's optional torchao 0.10 during Qwen loading.
# E45 uses NF4/FP16 and does not use torchao, so require its absence explicitly.
try:
    importlib.metadata.version("torchao")
except importlib.metadata.PackageNotFoundError:
    pass
else:
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True)
try:
    importlib.metadata.version("torchao")
except importlib.metadata.PackageNotFoundError:
    pass
else:
    raise AssertionError("torchao must be absent for frozen E45 runtime")

import faiss, numpy, sentence_transformers, torch
actual = {name: importlib.metadata.version(name) for name in PIP_PINS}
assert actual == PIP_PINS, (actual, PIP_PINS)
assert str(torch.__version__) == "2.10.0+cu128", torch.__version__
assert hasattr(faiss, "read_index") and hasattr(faiss, "IndexFlatIP"), "FAISS import lacks required APIs"
assert numpy.__version__ == PIP_PINS["numpy"]
assert sentence_transformers.__version__ == PIP_PINS["sentence-transformers"]
assert torch.cuda.is_available() and torch.cuda.device_count() == 2, "Select GPU T4 x2."
print({"runtime": {**actual, "torch": str(torch.__version__)},
       "gpus": [torch.cuda.get_device_name(index) for index in range(2)]}, flush=True)
'''
    bootstrap = r'''
from pathlib import Path
import hashlib, json, os, shutil, stat, sys, zipfile
INPUT, WORK = Path("/kaggle/input"), Path("/kaggle/working")
SHARD_INDEX, SHARD_COUNT = __SHARD__, 4
EXPECTED_SYSTEM_SHA256 = "__SYSTEM_SHA__"
def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
bin_inputs = sorted(path for path in INPUT.rglob("*.bin") if path.is_file())
print(f"[E45 private] Resolving system archive from {len(bin_inputs)} .bin inputs...", flush=True)
archives = []
for ordinal, path in enumerate(bin_inputs, start=1):
    digest = sha256(path)
    print({"system_archive_hash_progress": f"{ordinal}/{len(bin_inputs)}", "file": path.name, "bytes": path.stat().st_size}, flush=True)
    if digest == EXPECTED_SYSTEM_SHA256:
        archives.append(path)
assert len(archives) == 1, archives
system_bin = archives[0]
system_sidecar = Path(str(system_bin) + ".sha256")
assert system_sidecar.is_file(), "Attach the private system SHA-256 sidecar too."
assert system_sidecar.read_text(encoding="utf-8").strip().split()[0].lower() == EXPECTED_SYSTEM_SHA256, "Private system sidecar mismatch."
PROJECT_ROOT = WORK / "e45-private-system"
assert not PROJECT_ROOT.exists(), "Restart the kernel before re-running this cell."
PROJECT_ROOT.mkdir(parents=True)
with zipfile.ZipFile(system_bin) as archive:
    manifest = json.loads(archive.read("SYSTEM_FILE_MANIFEST.json").decode("utf-8"))
    archive_members = [item.filename for item in archive.infolist()]
    assert len(archive_members) == len(set(archive_members)), "duplicate system archive member"
    assert set(archive_members) == set(manifest) | {"SYSTEM_FILE_MANIFEST.json"}, "system member manifest mismatch"
    seen = set()
    for name, details in manifest.items():
        assert name and not name.startswith("/") and ".." not in name.split("/") and name.casefold() not in seen
        info = archive.getinfo(name); assert not stat.S_ISLNK(info.external_attr >> 16)
        data = archive.read(name); assert len(data) == details["bytes"] and hashlib.sha256(data).hexdigest() == details["sha256"]
        target = (PROJECT_ROOT / name).resolve(); assert target.is_relative_to(PROJECT_ROOT.resolve())
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data); seen.add(name.casefold())
assert (PROJECT_ROOT / "src/e45_private_shards.py").is_file()
sys.path.insert(0, str(PROJECT_ROOT / "src"))
'''.replace("__SHARD__", str(shard)).replace("__SYSTEM_SHA__", system_sha)
    intake = r'''
from e45_private_checkpoint import find_private_resume_checkpoint
from e45_private_shards import PRIVATE_SHA256, load_admission

admissions = [path for name in ("E45_PRIVATE_ADMISSION_R1.json", "E45_PRIVATE_DIRECT_RELEASE_R1.json") for path in INPUT.rglob(name)]
assert len(admissions) == 1, "Attach exactly one E45 release token."
ADMISSION = admissions[0]
admission = load_admission(ADMISSION)

# Hash every attached input exactly once.  This is deliberately visible: E00
# and E02 are several GB, and a silent hash scan looked like a stuck notebook.
all_input_files = sorted(path for path in INPUT.rglob("*") if path.is_file() and not path.is_symlink())
print(f"[E45 private] Verifying {len(all_input_files)} attached files by SHA-256...", flush=True)
hash_to_paths = {}
for ordinal, path in enumerate(all_input_files, start=1):
    digest = sha256(path)
    hash_to_paths.setdefault(digest, []).append(path)
    print({"input_hash_progress": f"{ordinal}/{len(all_input_files)}", "file": path.name, "bytes": path.stat().st_size}, flush=True)

def resolve_one(label, digest):
    matches = hash_to_paths.get(digest.lower(), [])
    assert len(matches) == 1, f"{label}: expected exactly one SHA-256 match, found {matches}"
    return matches[0]

def copy_verified_directory(destination, required):
    assert not destination.exists(), f"Restart kernel before recreating {destination.name}."
    destination.mkdir(parents=True)
    for relative, digest in required.items():
        source = resolve_one(relative, digest)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert sha256(target) == digest
    actual = {path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_file()}
    assert actual == set(required), (actual, set(required))
    return destination

# Resolve by content, never browser-renamed extension or Kaggle dataset slug.
candidate_source = resolve_one("Account A candidate", admission["candidate_archive_sha256"])
candidate_sidecars = [
    path for path in all_input_files if path.name.endswith(".sha256")
    and path.read_text(encoding="utf-8").strip().split()
    and path.read_text(encoding="utf-8").strip().split()[0].lower() == admission["candidate_archive_sha256"]
]
assert len(candidate_sidecars) == 1, "Attach exactly one matching candidate SHA-256 sidecar."
candidate_dir = WORK / "admitted-candidate"
CANDIDATE = copy_verified_directory(candidate_dir, {"E45_ACCOUNT_A_CANDIDATE.bin": admission["candidate_archive_sha256"]}) / "E45_ACCOUNT_A_CANDIDATE.bin"
Path(str(CANDIDATE) + ".sha256").write_text(f'{admission["candidate_archive_sha256"]}  {CANDIDATE.name}\n', encoding="ascii")
PRIVATE = resolve_one("official private questions", PRIVATE_SHA256)
E00 = {"manifest.json": "04efd3905ad6d2758461587ca68d7d70fa2c568855b78f0c762c7a37ba547b2e", "bm25.sqlite3": "f874a9528433f0db64efe7b3e951028d89433cbfb4c6e951a182b531acf1e0f0", "chunks.jsonl": "1e48c7762765ac2dd169045e9f5327c5311db3f1da8a6007ef70fff58718e367", "documents.jsonl": "f2968724e8a25124034b9ff2144427f8853ed37359443bd149ec3832f4c1fed7"}
E02 = {"manifest.json": "1b4d23c0149457055559ba1327b4dd5a1ae60bf07fa15a70b1e41d4ea514b270", "dense.faiss": "96ab6b8afcb376e327116642e0ffc378d633d9f712b698afd0e43dcee654a633", "chunk_ids.jsonl": "6e26962a0963f50460ada74707db31e604dfe4991e7d7150afeec89aa363fb99"}
e00_dir = copy_verified_directory(WORK / "verified-e00", E00)
e02_dir = copy_verified_directory(WORK / "verified-e02", E02)
RESUME_CHECKPOINT = find_private_resume_checkpoint(search_roots=[INPUT], shard_index=SHARD_INDEX, shard_count=SHARD_COUNT)
print({"shard": f"{SHARD_INDEX}/{SHARD_COUNT}", "candidate": str(CANDIDATE), "private": str(PRIVATE), "resume_checkpoint": str(RESUME_CHECKPOINT) if RESUME_CHECKPOINT else None}, flush=True)
'''
    execute = r'''
# Explicit imports keep this cell robust after a notebook-cell rerun.  The
# immutable inputs were already hash-resolved in the prior cell.
import json, os, subprocess, sys, time
from pathlib import Path
from transformers import AutoTokenizer

WALL_CLOCK_SAFETY_SECONDS = 39_000  # exit cleanly about 70 minutes before a 12h Kaggle limit
tokenizer_dir = WORK / "frozen-viqwen-tokenizer"
assert not tokenizer_dir.exists(), "Restart the kernel before rerunning generation setup."
print("[E45 private] Downloading pinned Vi-Qwen tokenizer, then starting answer-blind retrieval...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(
    "AITeamVN/Vi-Qwen2-3B-RAG",
    revision="eaf427c24d86066a2b35828c499b7db3af321227",
    trust_remote_code=False,
    fix_mistral_conversions=False,
)
tokenizer.save_pretrained(tokenizer_dir)
assert (tokenizer_dir / "tokenizer.json").is_file(), "Tokenizer save failed."
output = WORK / f"e45-private-shard-{SHARD_INDEX}-of-{SHARD_COUNT}"
assert not output.exists(), "Restart the kernel before rerunning shard generation."
checkpoint = WORK / f"E45_PRIVATE_SHARD_{SHARD_INDEX}_OF_{SHARD_COUNT}_CHECKPOINT.bin"
print({"event": "private_shard_start", "shard": f"{SHARD_INDEX}/4", "resume": bool(RESUME_CHECKPOINT), "wall_clock_safety_seconds": WALL_CLOCK_SAFETY_SECONDS}, flush=True)
run_command = [
    sys.executable, str(PROJECT_ROOT / "scripts/run_e45_private_shard.py"), "run",
    "--admission", str(ADMISSION), "--candidate-bin", str(CANDIDATE),
    "--private-questions", str(PRIVATE), "--e00", str(e00_dir), "--e02", str(e02_dir),
    "--tokenizer", str(tokenizer_dir), "--output", str(output),
    "--config", str(PROJECT_ROOT / "configs/e45-inference-aligned-parent-lora-v1.json"),
    "--shard-index", str(SHARD_INDEX), "--shard-count", str(SHARD_COUNT),
    "--checkpoint-archive", str(checkpoint), "--wall-clock-seconds", str(WALL_CLOCK_SAFETY_SECONDS),
]
if RESUME_CHECKPOINT is not None:
    run_command.extend(["--resume-checkpoint", str(RESUME_CHECKPOINT)])
# Keep an operator-visible heartbeat while retrieval and both isolated T4
# workers run. State files and the compact checkpoint archive update after
# every committed answer.
environment = {**os.environ, "PYTORCH_ALLOC_CONF": "expandable_segments:True"}
process = subprocess.Popen(run_command, env=environment)
started = time.monotonic()
last_heartbeat = -30
while process.poll() is None:
    elapsed = int(time.monotonic() - started)
    if elapsed - last_heartbeat >= 30:
        contexts_file = output / "contexts" / "holdout-contexts.jsonl"
        context_count = sum(1 for line in contexts_file.open(encoding="utf-8") if line.strip()) if contexts_file.is_file() else 0
        states = {}
        state_dir = output / "generation" / "worker-states"
        for state_file in sorted(state_dir.glob("*.json")) if state_dir.is_dir() else []:
            try:
                state = json.loads(state_file.read_text(encoding="utf-8"))
                states[state_file.name] = {"status": state.get("status"), "completed": len(state.get("completed", [])), "assigned": len(state.get("assigned_indices", [])), "second_pass_completed": len(state.get("second_pass_completed", [])), "device": state.get("device", state.get("device_map"))}
            except (OSError, json.JSONDecodeError):
                states[state_file.name] = {"status": "writing"}
        print({"private_heartbeat_seconds": elapsed, "contexts_ready": context_count, "checkpoint_exists": checkpoint.is_file(), "generation_workers": states}, flush=True)
        last_heartbeat = elapsed
    time.sleep(2)
if process.returncode != 0:
    raise subprocess.CalledProcessError(process.returncode, run_command)
execution_status = json.loads((output / "execution-status.json").read_text(encoding="utf-8"))
print({"event": "private_shard_segment_finished", "status": execution_status["status"], "completed_first_pass": execution_status["completed_first_pass"], "total": execution_status["total"], "checkpoint": str(checkpoint)}, flush=True)
'''
    package = r'''
bundle = WORK / f"E45_PRIVATE_SHARD_{SHARD_INDEX}_OF_{SHARD_COUNT}.bin"
checkpoint = WORK / f"E45_PRIVATE_SHARD_{SHARD_INDEX}_OF_{SHARD_COUNT}_CHECKPOINT.bin"
package_command = [
    sys.executable, str(PROJECT_ROOT / "scripts/run_e45_private_shard.py"), "package",
    "--admission", str(ADMISSION), "--candidate-bin", str(CANDIDATE),
    "--private-questions", str(PRIVATE), "--e00", str(e00_dir), "--e02", str(e02_dir),
    "--tokenizer", str(tokenizer_dir), "--output", str(output),
    "--config", str(PROJECT_ROOT / "configs/e45-inference-aligned-parent-lora-v1.json"),
    "--shard-index", str(SHARD_INDEX), "--shard-count", str(SHARD_COUNT),
    "--checkpoint-archive", str(checkpoint), "--archive", str(bundle),
]
subprocess.run(package_command, check=True)
if bundle.is_file():
    print({"status": "COMPLETE", "bundle": str(bundle), "sidecar": str(Path(str(bundle) + ".sha256"))}, flush=True)
else:
    print({"status": "CHECKPOINTED", "checkpoint": str(checkpoint), "sidecar": str(Path(str(checkpoint) + ".sha256")), "next": "Attach this Output dataset to the next session for the same shard."}, flush=True)
'''
    return {"cells": [_cell(markdown, "markdown"), _cell(environment), _cell(bootstrap), _cell(intake), _cell(execute), _cell(package)], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}, "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True, "isInternetEnabled": True, "language": "python", "sourceType": "notebook"}}, "nbformat": 4, "nbformat_minor": 5}


def main() -> int:
    digest = build_system()
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    for shard in range(4):
        path = NOTEBOOKS / f"E45-PRIVATE-RESUMABLE-SHARD-{shard}-OF-4-T4X2.ipynb"
        path.write_text(json.dumps(_notebook(shard, digest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"system": str(SYSTEM), "sha256": digest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
