"""Canonical E45 holdout context preparation pipeline using frozen P00/E44 retrieval and P01 prompt rendering."""

from __future__ import annotations

import hashlib
import gc
import json
import logging
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import e21_parent_context as parent
from .bm25 import SqliteBm25Index
from .corpus import file_sha256
from .e02_answer import _atomic_json, _atomic_jsonl
from .e02_compare import _ChunkStore, _TorchGpuFlatIpIndex, _load_mapping, weighted_rrf
from .e07_generator_ab import _as_text_chat_messages
from .e18_source_metadata import enrich_context, scan_selected
from .final_private_p00 import article_blocks, seed_unit
from .final_private_p01 import _prompt, _tokenizer, Config, _PromptRoot, _ContextConfig

LOG = logging.getLogger("e45_holdout_contexts")

PINNED_HOLDOUT_SAMPLE_IDS_SHA256 = "36c6faad7e7d52030ea98b00102281c707a798e33845a7ddb89f96473717d4d1"

PINNED_HOLDOUT_SAMPLE_IDS: tuple[str, ...] = (
    "9281", "49039", "33233", "6373", "39819", "158333", "144343", "86395", "38731", "121413",
    "124251", "137389", "39115", "95967", "66937", "160381", "164503", "125909", "61453", "44797",
    "47391", "47285", "124759", "7101", "119005", "39835", "20881", "55985", "111673", "2239",
    "144721", "139415", "128069", "127339", "163585", "142265", "60233", "25603", "87167", "105303",
    "164621", "164495", "105647", "81801", "117887", "85891", "134123", "146415", "88099", "163117",
    "149723", "111159", "139827", "29509", "118005", "68713", "44197", "112659", "15959", "100999",
    "132225", "124781", "167691", "78633", "85819", "80691", "103799", "15827", "36601", "22397",
    "141985", "11597", "33209", "38689", "25189", "67903", "93523", "141653", "90121", "93795",
    "53467", "11473", "4331", "33059", "97453", "94317", "72341", "61523", "56725", "89395",
    "91325", "103229", "119369", "150025", "101923", "121967", "136285", "109315", "66375", "106133",
    "29871", "152819", "160521", "1707", "162825", "83451", "70593", "25571", "87917", "11641",
    "166063", "101195", "82917", "140865", "126821", "92743", "145577", "118539", "89135", "125205",
    "162201", "19387", "77749", "63753", "125655", "82001", "20429", "157241", "30913", "96299",
    "33839", "15325", "88439", "17953", "116731", "44569", "133689", "22299", "112473", "87115",
    "155689", "127465", "33647", "108439", "154585", "23021", "47053", "19491", "92539", "102153",
    "80155", "5975", "155405", "104673", "76777", "143185", "31977", "122721", "161619", "149483",
    "138689", "155873", "100823", "47885", "156645", "130355", "160163", "148353", "108089", "59843",
    "104713", "54173", "127487", "115973", "61469", "78779", "151233", "68859", "121261", "72461",
    "34159", "2385", "158893", "71925", "13703", "93983", "68975", "106323", "85317", "156579",
    "29929", "17691", "102549", "4845", "59143", "111137", "71777", "62115", "36489", "12425",
)


class HoldoutError(Exception):
    """Raised when holdout preparation fails or invariants are violated."""


def normalize_question(q: str) -> str:
    return " ".join(unicodedata.normalize("NFC", q).strip().lower().split())


def compute_json_sha256(data: Any) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _contains_reference_field(value: Any) -> bool:
    """Reject reference-bearing keys anywhere in a reusable context record."""
    forbidden = {
        "answer", "answers", "reference", "references", "reference_answer",
        "gold_answer", "gold_evidence", "target", "targets",
    }
    if isinstance(value, dict):
        return any(
            str(key).casefold() in forbidden or _contains_reference_field(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_reference_field(item) for item in value)
    return False


def _validate_reusable_context_rows(
    *,
    rows: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    tokenizer: Any,
) -> None:
    """Fully validate a completed JSONL left by a failure after atomic write."""
    expected_fields = {
        "sample_index", "question_id", "question", "question_sha256",
        "ordered_seed_ids", "evidence_sha256", "prompt", "prompt_sha256",
        "prompt_input_ids_sha256", "input_tokens", "units", "answers_used",
        "record_sha256",
    }
    if len(rows) != len(questions) or len(rows) != 200:
        raise HoldoutError("Reusable holdout JSONL does not contain exactly 200 rows.")
    expected_ids = [str(item["question_id"]) for item in questions]
    observed_ids = [str(row.get("question_id", "")) for row in rows]
    if observed_ids != expected_ids or len(set(observed_ids)) != len(observed_ids):
        raise HoldoutError("Reusable holdout JSONL has wrong, duplicate, or reordered QIDs.")

    for index, (row, question) in enumerate(zip(rows, questions)):
        if set(row) != expected_fields:
            raise HoldoutError(f"Reusable context row {index} has an unexpected schema.")
        if row["sample_index"] != index or row["answers_used"] is not False:
            raise HoldoutError(f"Reusable context row {index} has invalid index/reference state.")
        payload_without_flag = {key: value for key, value in row.items() if key != "answers_used"}
        if _contains_reference_field(payload_without_flag):
            raise HoldoutError(f"Reusable context row {index} contains a reference-bearing field.")
        if row["question"] != question["question"]:
            raise HoldoutError(f"Reusable context row {index} question bytes changed.")
        if row["question_sha256"] != hashlib.sha256(row["question"].encode("utf-8")).hexdigest():
            raise HoldoutError(f"Reusable context row {index} question hash changed.")
        if len(row["ordered_seed_ids"]) != 12 or len(row["units"]) != 12:
            raise HoldoutError(f"Reusable context row {index} does not preserve 12 seeds.")
        if row["evidence_sha256"] != compute_json_sha256(
            {"ordered_seed_ids": row["ordered_seed_ids"], "units": row["units"]}
        ):
            raise HoldoutError(f"Reusable context row {index} evidence hash changed.")
        if row["prompt_sha256"] != hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest():
            raise HoldoutError(f"Reusable context row {index} prompt hash changed.")
        input_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        input_ids_sha = hashlib.sha256(
            json.dumps(input_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if row["prompt_input_ids_sha256"] != input_ids_sha or row["input_tokens"] != len(input_ids):
            raise HoldoutError(f"Reusable context row {index} tokenizer identity changed.")
        body = dict(row)
        observed_record_sha = body.pop("record_sha256")
        if observed_record_sha != compute_json_sha256(body):
            raise HoldoutError(f"Reusable context row {index} record hash changed.")


def _build_holdout_manifest(
    *,
    cfg: Config,
    contexts_rows: list[dict[str, Any]],
    out_jsonl: Path,
    tokenizer_path: Path,
    e00_cfg: dict[str, Any],
    dense_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Build the manifest from validated persisted rows and P01 config bytes."""
    return {
        "schema_version": "1.0",
        "experiment_id": cfg.raw["experiment_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_size": len(contexts_rows),
        "sample_ids_sha256": PINNED_HOLDOUT_SAMPLE_IDS_SHA256,
        "e00_manifest_sha256": e00_cfg["manifest_sha256"],
        "e02_manifest_sha256": dense_cfg.get("manifest_sha256", "1b4d23c0149457055559ba1327b4dd5a1ae60bf07fa15a70b1e41d4ea514b270"),
        "dense_model": dense_cfg["model_id"],
        "dense_model_revision": dense_cfg["revision"],
        "tokenizer_sha256": file_sha256(tokenizer_path / "tokenizer.json"),
        "config_sha256": cfg.sha,
        "contexts_jsonl_sha256": file_sha256(out_jsonl),
        "mean_input_tokens": float(np.mean([row["input_tokens"] for row in contexts_rows])),
        "answers_used": False,
    }


def select_group_safe_holdout(
    *,
    official_train_path: Path,
    official_warmup_path: Path,
    official_public_path: Path,
    splits_train_path: Path,
    splits_dev_path: Path,
    config: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deterministically reproduce the 200-row group-safe holdout strictly answer-blind."""
    train_split = json.loads(splits_train_path.read_text(encoding="utf-8"))
    dev_split = json.loads(splits_dev_path.read_text(encoding="utf-8"))
    excluded_ids = set(train_split.keys()) | set(dev_split.keys())

    excluded_questions = {
        normalize_question(v["question"]) for v in train_split.values()
    } | {
        normalize_question(v["question"]) for v in dev_split.values()
    }

    if official_public_path.is_file():
        pub = json.loads(official_public_path.read_text(encoding="utf-8"))
        excluded_questions |= {normalize_question(v["question"]) for v in pub.values()}

    records_pool: dict[str, dict[str, Any]] = {}
    if official_train_path.is_file():
        for qid, rec in json.loads(official_train_path.read_text(encoding="utf-8")).items():
            records_pool[str(qid)] = {"question": rec["question"]}
    if official_warmup_path.is_file():
        for qid, rec in json.loads(official_warmup_path.read_text(encoding="utf-8")).items():
            if str(qid) not in records_pool:
                records_pool[str(qid)] = {"question": rec["question"]}

    eligible_candidates: list[dict[str, Any]] = []
    for qid, rec in sorted(records_pool.items()):
        if qid in excluded_ids:
            continue
        nq = normalize_question(rec["question"])
        if nq in excluded_questions:
            continue
        eligible_candidates.append({
            "question_id": qid,
            "question": rec["question"],
            "normalized_question": nq,
        })

    groups: dict[str, list[dict[str, Any]]] = {}
    for item in eligible_candidates:
        groups.setdefault(item["normalized_question"], []).append(item)

    holdout_cfg = config.section("holdout") if hasattr(config, "section") else config["holdout"]
    salt = holdout_cfg.get("group_salt", "e34-long-context-holdout-v1\u0000")

    def group_sort_key(nq: str) -> str:
        return hashlib.sha256(salt.encode("utf-8") + nq.encode("utf-8")).hexdigest()

    sorted_group_keys = sorted(groups.keys(), key=group_sort_key)
    target_count = holdout_cfg["sample_size"]
    selected_records: list[dict[str, Any]] = []
    selected_groups: list[str] = []

    for nq in sorted_group_keys:
        group_items = sorted(groups[nq], key=lambda x: x["question_id"])
        if len(selected_records) + len(group_items) > target_count:
            continue
        selected_records.extend(group_items)
        selected_groups.append(nq)
        if len(selected_records) == target_count:
            break

    actual_ids = [r["question_id"] for r in selected_records]
    actual_ids_sha = hashlib.sha256("\n".join(actual_ids).encode("utf-8")).hexdigest()

    if len(selected_records) != target_count:
        raise HoldoutError(f"Holdout selection produced {len(selected_records)} records, expected {target_count}")
    if actual_ids_sha != PINNED_HOLDOUT_SAMPLE_IDS_SHA256:
        raise HoldoutError(f"Holdout sample IDs SHA {actual_ids_sha} != expected {PINNED_HOLDOUT_SAMPLE_IDS_SHA256}")

    identity = {
        "sample_size": len(selected_records),
        "group_count": len(selected_groups),
        "sample_ids_sha256": actual_ids_sha,
        "sample_ids": actual_ids,
    }
    return selected_records, identity


def _load_dense_searcher(dense_dir: Path, dense_cfg: dict[str, Any], device: str) -> Any:
    """Load FAISS index via _TorchGpuFlatIpIndex on CUDA or CPU FAISS fallback."""
    faiss_path = dense_dir / "dense.faiss"
    expected_count = dense_cfg["record_count"]
    expected_dimension = dense_cfg["dimension"]

    import torch
    if device.startswith("cuda:") and torch.cuda.is_available():
        return _TorchGpuFlatIpIndex(
            faiss_path,
            device=device,
            expected_count=expected_count,
            expected_dimension=expected_dimension,
        )

    # CPU searcher fallback using faiss or numpy
    try:
        import faiss
        index = faiss.read_index(str(faiss_path))
        if index.ntotal != expected_count or index.d != expected_dimension:
            raise HoldoutError("Loaded FAISS shape differs from pinned artifact.")

        class _FaissCpuIndex:
            def __init__(self, idx: Any):
                self._idx = idx

            def search(self, vectors: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
                return self._idx.search(vectors, top_k)

            def close(self) -> None:
                pass

        return _FaissCpuIndex(index)
    except ImportError:
        raise HoldoutError("faiss or torch with CUDA is required for dense search.")


def prepare_holdout_contexts(
    *,
    questions: list[dict[str, Any]],
    e00_dir: Path,
    dense_dir: Path,
    tokenizer_path: Path,
    output_dir: Path,
    config: Any,
    device: str = "cuda:0",
) -> tuple[Path, dict[str, Any]]:
    """Execute canonical P00 retrieval and P01 prompt rendering on holdout questions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config(config.raw, config.path) if not isinstance(config, Config) else config

    # 1. Validate artifact manifests
    e00_manifest = json.loads((e00_dir / "manifest.json").read_text(encoding="utf-8"))
    dense_manifest = json.loads((dense_dir / "manifest.json").read_text(encoding="utf-8"))

    e00_cfg = cfg.section("source_e00")
    dense_cfg = cfg.section("source_dense")
    retrieval_cfg = cfg.section("retrieval")

    if file_sha256(e00_dir / "manifest.json") != e00_cfg["manifest_sha256"]:
        raise HoldoutError("E00 manifest hash mismatch.")
    if file_sha256(dense_dir / "manifest.json") != dense_cfg.get("manifest_sha256", "1b4d23c0149457055559ba1327b4dd5a1ae60bf07fa15a70b1e41d4ea514b270"):
        raise HoldoutError("E02 manifest hash mismatch.")

    ids = [q["question_id"] for q in questions]
    q_map = {q["question_id"]: q["question"] for q in questions}

    # The JSONL is written atomically before its manifest. If a prior run was
    # interrupted after that rename, validate every record and finalize it
    # without repeating BM25/dense retrieval.
    out_jsonl = output_dir / "holdout-contexts.jsonl"
    if out_jsonl.is_file():
        LOG.info("Found existing holdout JSONL; validating it for fail-closed recovery...")
        tokenizer = _tokenizer(tokenizer_path)
        try:
            contexts_rows = [
                json.loads(line)
                for line in out_jsonl.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError) as exc:
            raise HoldoutError("Existing holdout JSONL is unreadable or invalid.") from exc
        _validate_reusable_context_rows(
            rows=contexts_rows,
            questions=questions,
            tokenizer=tokenizer,
        )
        manifest = _build_holdout_manifest(
            cfg=cfg,
            contexts_rows=contexts_rows,
            out_jsonl=out_jsonl,
            tokenizer_path=tokenizer_path,
            e00_cfg=e00_cfg,
            dense_cfg=dense_cfg,
        )
        _atomic_json(output_dir / "holdout-manifest.json", manifest)
        LOG.info(
            "Recovered validated holdout contexts without rerunning retrieval (SHA-256: %s)",
            manifest["contexts_jsonl_sha256"],
        )
        return out_jsonl, manifest

    # 2. Sparse BM25 retrieval
    LOG.info("Running sparse BM25 retrieval for %d questions...", len(ids))
    bm25_path = e00_dir / "bm25.sqlite3"
    sparse_top_ids: dict[str, list[str]] = {}
    bm25_index = SqliteBm25Index(bm25_path)
    try:
        for index, qid in enumerate(ids, start=1):
            hits = bm25_index.search(q_map[qid], top_k=retrieval_cfg["candidate_k_per_branch"])
            # SqliteBm25Index returns immutable Bm25Hit objects, as does frozen
            # P00. Accessing them as mappings breaks before hybrid retrieval.
            sparse_top_ids[qid] = [hit.chunk_id for hit in hits]
            if index == 1 or index % 25 == 0 or index == len(ids):
                LOG.info("holdout_sparse_progress completed=%d total=%d", index, len(ids))
    finally:
        bm25_index.close()

    # 3. Open and shape-check the pinned FAISS artifact before downloading the
    # embedding model. This makes missing/binary-incompatible FAISS fail early.
    LOG.info("Loading and validating pinned FAISS index before embedding model download...")
    searcher = _load_dense_searcher(dense_dir, dense_cfg, device)
    mapping = _load_mapping(dense_dir / "chunk_ids.jsonl", dense_cfg["record_count"])
    store = _ChunkStore(bm25_path)

    dense_top_ids: dict[str, list[str]] = {}
    fused_results: dict[str, list[dict[str, Any]]] = {}
    raw_contexts: dict[str, list[dict[str, Any]]] = {}

    batch_size = 32
    query_prefix = dense_cfg.get("query_prefix", "")
    dense_model: Any | None = None
    try:
        LOG.info("Loading dense model %s (revision: %s)...", dense_cfg["model_id"], dense_cfg["revision"])
        from sentence_transformers import SentenceTransformer
        dense_model = SentenceTransformer(
            dense_cfg["model_id"], revision=dense_cfg["revision"], device=device
        )
        for start in range(0, len(ids), batch_size):
            batch_qids = ids[start : start + batch_size]
            query_texts = [query_prefix + q_map[qid] for qid in batch_qids]
            vectors = dense_model.encode(
                query_texts,
                batch_size=len(batch_qids),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            vectors = np.asarray(vectors, dtype=np.float32)
            scores, rows = searcher.search(vectors, retrieval_cfg["candidate_k_per_branch"])

            for offset, qid in enumerate(batch_qids):
                d_ids = [mapping[int(r)] for r in rows[offset] if int(r) >= 0]
                dense_top_ids[qid] = d_ids

                fused = weighted_rrf(
                    {"sparse": sparse_top_ids[qid], "dense": d_ids},
                    weights={"sparse": retrieval_cfg["sparse_weight"], "dense": retrieval_cfg["dense_weight"]},
                    constant=retrieval_cfg["rrf_constant"],
                    top_k=retrieval_cfg["fused_top_k"],
                )
                fused_results[qid] = fused
                top12 = store.fetch([item["chunk_id"] for item in fused[:retrieval_cfg["selected_contexts"]]])
                if len(top12) != 12:
                    raise HoldoutError(f"Expected 12 contexts for {qid}, got {len(top12)}")
                raw_contexts[qid] = top12
            LOG.info(
                "holdout_dense_progress completed=%d total=%d",
                min(start + len(batch_qids), len(ids)),
                len(ids),
            )
    finally:
        searcher.close()
        store.close()
        if dense_model is not None:
            del dense_model
        gc.collect()
        if device.startswith("cuda:"):
            import torch
            torch.cuda.empty_cache()
        LOG.info("Released dense retrieval model and index memory.")

    # 4. Parent expansion using article_blocks and seed_unit
    LOG.info("Running parent expansion for %d questions...", len(ids))
    metadata_cfg = cfg.section("metadata_source")
    wanted_cids = {ctx["chunk_id"] for top12 in raw_contexts.values() for ctx in top12}
    seeds = scan_selected(e00_dir / metadata_cfg["chunks_path"], "chunk_id", wanted_cids, metadata_cfg["chunks_sha256"])
    doc_ids = {s["document_id"] for s in seeds.values()}
    docs = scan_selected(e00_dir / metadata_cfg["documents_path"], "document_id", doc_ids, metadata_cfg["documents_sha256"])

    by_doc: dict[str, list[dict[str, Any]]] = {d_id: [] for d_id in doc_ids}
    with (e00_dir / metadata_cfg["chunks_path"]).open("rb") as stream:
        for line in stream:
            chunk = json.loads(line)
            if chunk["document_id"] in by_doc:
                by_doc[chunk["document_id"]].append(chunk)

    block_by_chunk: dict[str, list[dict[str, Any]]] = {}
    for doc_id, chunks in by_doc.items():
        for block in article_blocks(chunks, docs[doc_id]):
            for chunk in block:
                if chunk["chunk_id"] in wanted_cids:
                    block_by_chunk[chunk["chunk_id"]] = block

    prepared_units: dict[str, list[dict[str, Any]]] = {}
    for qid in ids:
        units = []
        for rank, ctx in enumerate(raw_contexts[qid]):
            seed = seeds[ctx["chunk_id"]]
            enrich_context(ctx, seed, docs[seed["document_id"]])
            units.append(seed_unit(seed, rank, block_by_chunk[seed["chunk_id"]], docs[seed["document_id"]], cfg.section("context_policy")))
        prepared_units[qid] = units

    # 5. Render prompts using canonical P01 _prompt
    LOG.info("Rendering prompts with Vi-Qwen tokenizer...")
    tokenizer = _tokenizer(tokenizer_path)

    contexts_rows: list[dict[str, Any]] = []
    for idx, qid in enumerate(ids):
        prep_row = {"question_id": qid, "sample_index": idx, "answers_used": False, "units": prepared_units[qid]}
        prompt_str, tokens_count, spans, packing, _ = _prompt(tokenizer, q_map[qid], prep_row, cfg)
        prompt_input_ids = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
        ordered_seed_ids = [ctx["chunk_id"] for ctx in raw_contexts[qid]]
        evidence_sha256 = compute_json_sha256(
            {"ordered_seed_ids": ordered_seed_ids, "units": prepared_units[qid]}
        )

        record_body = {
            "sample_index": idx,
            "question_id": qid,
            "question": q_map[qid],
            "question_sha256": hashlib.sha256(q_map[qid].encode("utf-8")).hexdigest(),
            "ordered_seed_ids": ordered_seed_ids,
            "evidence_sha256": evidence_sha256,
            "prompt": prompt_str,
            "prompt_sha256": hashlib.sha256(prompt_str.encode("utf-8")).hexdigest(),
            "prompt_input_ids_sha256": hashlib.sha256(
                json.dumps(prompt_input_ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "input_tokens": tokens_count,
            "units": prepared_units[qid],
            "answers_used": False,
        }
        record_body["record_sha256"] = compute_json_sha256(record_body)
        if idx == 0 or (idx + 1) % 25 == 0 or idx + 1 == len(ids):
            LOG.info("holdout_prompt_progress completed=%d total=%d", idx + 1, len(ids))
        contexts_rows.append(record_body)

    _atomic_jsonl(out_jsonl, contexts_rows)
    manifest = _build_holdout_manifest(
        cfg=cfg,
        contexts_rows=contexts_rows,
        out_jsonl=out_jsonl,
        tokenizer_path=tokenizer_path,
        e00_cfg=e00_cfg,
        dense_cfg=dense_cfg,
    )
    manifest_path = output_dir / "holdout-manifest.json"
    _atomic_json(manifest_path, manifest)
    LOG.info("Holdout contexts written to %s (SHA-256: %s)", out_jsonl, manifest["contexts_jsonl_sha256"])
    return out_jsonl, manifest
