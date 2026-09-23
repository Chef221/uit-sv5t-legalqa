"""E45 canonical inference-aligned parent-context training preparation and dataset."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Callable

from . import e21_parent_context as parent
from .corpus import file_sha256
from .e02_answer import _atomic_json, _atomic_jsonl, make_messages
from .e07_generator_ab import _as_text_chat_messages
from .e08b_context_lora import inference_packing
from .e18_source_metadata import enrich_context, scan_selected

LOG = logging.getLogger(__name__)

EXPERIMENT_ID = "E45-inference-aligned-parent-lora-v1"
PINNED_TRAINING_IDS_SHA256 = "fee946ada7cbfca7df04ffe1c3d9f06907808c0dc59600e6f1f8d579076742dd"
PINNED_E38_ADAPTER_SHA256 = "e8f55f088fe2c5951c336095a5682af3a53f62c8be186bd5bfbf34741d895222"


class E45Error(RuntimeError):
    """Raised when E45 contract invariants are violated."""


def normalize_question(text: str) -> str:
    """Deterministic question text normalization."""
    text = unicodedata.normalize("NFKC", text)
    return " ".join(text.lower().split())


@dataclass(frozen=True)
class E45Config:
    raw: dict[str, Any]
    path: Path
    sha256: str

    @property
    def experiment_id(self) -> str:
        return self.raw.get("experiment_id", EXPERIMENT_ID)

    def section(self, key: str) -> dict[str, Any]:
        if key not in self.raw:
            raise E45Error(f"Missing required config section: {key}")
        return self.raw[key]


def load_config(path: Path) -> E45Config:
    """Load and validate E45 frozen configuration."""
    if not path.is_file():
        raise E45Error(f"E45 config not found: {path}")
    data = path.read_bytes()
    raw = json.loads(data.decode("utf-8"))
    sha = hashlib.sha256(data).hexdigest()
    cfg = E45Config(raw=raw, path=path, sha256=sha)
    validate_config(cfg)
    return cfg


def validate_config(config: E45Config) -> None:
    """Strictly assert all quality-affecting config fields match contract."""
    raw = config.raw
    if raw.get("schema_version") != "1.0":
        raise E45Error("Config schema_version must be '1.0'")
    if raw.get("experiment_id") != EXPERIMENT_ID:
        raise E45Error(f"Invalid experiment_id: {raw.get('experiment_id')}")

    model_inv = raw.get("model_inventory", {})
    if model_inv.get("base_model") != "AITeamVN/Vi-Qwen2-3B-RAG":
        raise E45Error("Base model must be 'AITeamVN/Vi-Qwen2-3B-RAG'")
    if model_inv.get("base_revision") != "eaf427c24d86066a2b35828c499b7db3af321227":
        raise E45Error("Base revision must be pinned eaf427c24d86066a2b35828c499b7db3af321227")
    if model_inv.get("expected_lora_trainable_parameters") != 14966784:
        raise E45Error("Expected LoRA trainable parameters must be 14,966,784")
    if model_inv.get("expected_actual_stack_total") != 3668660224:
        raise E45Error("Expected actual stack total must be 3,668,660,224")
    if model_inv.get("exclusive_competition_limit") != 4000000000:
        raise E45Error("Exclusive competition limit must be 4,000,000,000")

    training = raw.get("training", {})
    if training.get("records_count") != 5636:
        raise E45Error("Training records_count must be 5636")
    if training.get("training_sample_ids_sha256") != PINNED_TRAINING_IDS_SHA256:
        raise E45Error("Training sample-IDs hash mismatch")
    if training.get("epochs") != 1.0:
        raise E45Error("Training epochs must be 1.0")
    if training.get("expected_optimizer_steps") != 705:
        raise E45Error("Expected optimizer steps must be 705")
    if training.get("lora_rank") != 8 or training.get("lora_alpha") != 16:
        raise E45Error("LoRA rank must be 8 and alpha 16")
    required_training = {
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "seed": 20260830,
        "learning_rate": 1e-4,
        "per_device_train_batch": 1,
        "gradient_accumulation": 4,
        "world_size": 2,
        "effective_global_batch": 8,
        "warmup_ratio": 0.03,
        "scheduler": "cosine",
        "weight_decay": 0.0,
        "save_interval_steps": 32,
        "quantization": "nf4-double-quant",
        "compute_dtype": "float16",
        "gradient_checkpointing": True,
        "loss": "assistant-target-only",
        "target": "complete-official-answer-followed-by-eos",
    }
    for field, expected in required_training.items():
        if training.get(field) != expected:
            raise E45Error(f"Frozen training field changed: {field}")
    if training.get("maximum_total_sequence") != 8192:
        raise E45Error("Maximum total sequence must be 8192")
    if training.get("effective_global_batch") != 8:
        raise E45Error("Effective global batch must be 8")

    ctx = raw.get("context_policy", {})
    if ctx.get("seed_contexts") != 12:
        raise E45Error("Seed contexts must be 12")
    if ctx.get("max_parent_tokens") != 1200 or ctx.get("max_parent_characters") != 6000:
        raise E45Error("Context policy parent bounds altered")
    if ctx.get("max_input_tokens") != 8192:
        raise E45Error("Context policy max_input_tokens must be 8192")
    for field, expected in {
        "neighbor_radius": 1,
        "expansion_order": ["whole_article", "both_neighbors", "previous_neighbor", "next_neighbor"],
        "evict_seed_for_expansion": False,
        "merge": "overlapping-or-touching-exact-document-spans",
    }.items():
        if ctx.get(field) != expected:
            raise E45Error(f"Frozen context field changed: {field}")

    retrieval = raw.get("retrieval", {})
    for field, expected in {
        "candidate_k_per_branch": 40,
        "rrf_constant": 60,
        "sparse_weight": 0.5,
        "dense_weight": 0.5,
        "fused_top_k": 20,
        "selected_contexts": 12,
        "normalize_query_embeddings": True,
        "reranker": None,
    }.items():
        if retrieval.get(field) != expected:
            raise E45Error(f"Frozen retrieval field changed: {field}")

    inference = raw.get("inference", {})
    for field, expected in {
        "precision": "float16",
        "greedy_decoding": True,
        "sampling": False,
        "initial_max_new_tokens": 1024,
        "second_pass_max_new_tokens": 1536,
        "clean_up_tokenization_spaces": False,
        "replica_max_input_tokens": 6000,
        "maximum_fixed_point_passes_per_stage": 16,
        "suffix_only": True,
        "interior_trim": False,
    }.items():
        if inference.get(field) != expected:
            raise E45Error(f"Frozen inference field changed: {field}")

    runtime = raw.get("runtime", {})
    expected_runtime = {
        "torch": "2.10.0+cu128",
        "transformers": "5.16.1",
        "peft": "0.19.1",
        "accelerate": "1.13.0",
        "bitsandbytes": "0.50.2",
        "sentence-transformers": "5.4.1",
        "faiss-cpu": "1.15.0",
        "numpy": "2.0.2",
    }
    if runtime != expected_runtime:
        raise E45Error("Frozen runtime contract changed")

    control = raw.get("control", {})
    if control.get("adapter_sha256") != PINNED_E38_ADAPTER_SHA256:
        raise E45Error("Control adapter hash mismatch")


class _E45PromptRoot:
    def __init__(self, config: E45Config):
        self.config = config

    def section(self, key: str) -> dict[str, Any]:
        if key == "inference":
            return {"max_input_tokens": 8192, "minimum_contexts": 1}
        if key == "prompt":
            return self.config.section("prompt")
        raise E45Error(f"Unexpected prompt section: {key}")


class _E45ContextSource:
    def __init__(self, config: E45Config):
        self.source = _E45PromptRoot(config)


class _E45ContextConfig:
    def __init__(self, config: E45Config, policy: dict[str, Any]):
        self.policy = policy
        self.source = _E45ContextSource(config)


def render_canonical_prompt(messages: list[dict[str, Any]], tokenizer: Any) -> str:
    """Render chat template using canonical Qwen2 text messages."""
    return tokenizer.apply_chat_template(
        _as_text_chat_messages(messages), tokenize=False, add_generation_prompt=True
    )


def count_text_tokens(text: str, tokenizer: Any) -> int:
    """Count tokens without special tokens."""
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def pack_training_evidence(
    *,
    question: str,
    units: list[dict[str, Any]],
    answer_tokens_count: int,
    tokenizer: Any,
    config: E45Config,
) -> tuple[str, int, list[dict[str, Any]], dict[str, Any]]:
    """Pack parent-expanded context respecting 8,192 token limit and preserving all 12 seeds.

    1. Reserve answer tokens + EOS first.
    2. Budget = 8,192 - answer_tokens_count.
    3. Assert all 12 compact seeds fit within budget.
    4. Run parent expansion in rank order (0 to 11).
    5. If total prompt tokens exceed budget, drop parent expansion from lowest-priority rank end (11 down to 0).
    """
    total_budget = config.section("training")["maximum_total_sequence"]
    prompt_budget = total_budget - answer_tokens_count
    if prompt_budget <= 0:
        raise E45Error(f"Answer length {answer_tokens_count} exceeds total sequence budget {total_budget}")

    policy = copy.deepcopy(config.section("context_policy"))
    policy["max_input_tokens"] = prompt_budget

    context_cfg = _E45ContextConfig(config, policy)
    packing_cfg = inference_packing(context_cfg.source.source)

    def msg_counter(msgs: list[dict[str, Any]]) -> int:
        prompt_str = render_canonical_prompt(msgs, tokenizer)
        return count_text_tokens(prompt_str, tokenizer)

    def txt_counter(text: str) -> int:
        return count_text_tokens(text, tokenizer)

    # 1. First verify compact seeds fit
    chosen_compact: list[dict[str, Any]] = []
    for u in units:
        val = {k: v for k, v in u.items() if k not in ("seed", "expansions")}
        val.update(u["seed"])
        chosen_compact.append(val)
    compact_msgs, compact_merged = parent.render(question, chosen_compact, packing_cfg)
    compact_tokens = msg_counter(compact_msgs)
    if compact_tokens > prompt_budget:
        raise E45Error(
            f"All 12 compact seeds ({compact_tokens} tokens) cannot fit inside remaining prompt budget ({prompt_budget} tokens)"
        )

    # 2. Run E21 parent expansion with prompt_budget
    messages, merged, diagnostics = parent.pack(
        question,
        {"units": units},
        context_cfg,
        msg_counter,
        txt_counter,
        parent.VARIANTS[1],
    )

    rendered_prompt = render_canonical_prompt(messages, tokenizer)
    prompt_tokens = count_text_tokens(rendered_prompt, tokenizer)

    # 3. Fallback: if prompt_tokens > prompt_budget, drop expansions from lowest-priority rank end
    if prompt_tokens > prompt_budget:
        # Revert expansions starting from highest seed rank (lowest priority)
        active_units = copy.deepcopy(units)
        # Find which seeds were expanded
        expanded_ranks = sorted([action["seed_rank"] for action in diagnostics.get("expansions", [])], reverse=True)
        for rank_to_revert in expanded_ranks:
            # Revert this rank to compact seed
            reverted_chosen: list[dict[str, Any]] = []
            for u in active_units:
                val = {k: v for k, v in u.items() if k not in ("seed", "expansions")}
                if u["rank"] == rank_to_revert:
                    val.update(u["seed"])
                else:
                    # check if u was previously expanded
                    matching_action = [a for a in diagnostics.get("expansions", []) if a["seed_rank"] == u["rank"]]
                    if matching_action and u["expansions"]:
                        ext = next((e for e in u["expansions"] if e["kind"] == matching_action[0]["kind"]), None)
                        if ext:
                            val.update({k: v for k, v in ext.items() if k != "kind"})
                        else:
                            val.update(u["seed"])
                    else:
                        val.update(u["seed"])
                reverted_chosen.append(val)
            trial_msgs, trial_merged = parent.render(question, reverted_chosen, packing_cfg)
            trial_prompt = render_canonical_prompt(trial_msgs, tokenizer)
            trial_tokens = count_text_tokens(trial_prompt, tokenizer)
            if trial_tokens <= prompt_budget:
                messages = trial_msgs
                merged = trial_merged
                rendered_prompt = trial_prompt
                prompt_tokens = trial_tokens
                diagnostics["expansions"] = [a for a in diagnostics["expansions"] if a["seed_rank"] != rank_to_revert]
                break

    if prompt_tokens + answer_tokens_count > total_budget:
        raise E45Error(
            f"Prompt ({prompt_tokens}) + Answer ({answer_tokens_count}) = {prompt_tokens + answer_tokens_count} > {total_budget}"
        )

    # Assert all 12 seed ranks are represented in merged spans
    observed_ranks = sorted({u["rank"] for u in merged})
    if len(diagnostics.get("seed_ranks", [])) != 12 or diagnostics.get("skipped_seed_ranks"):
        raise E45Error(f"Lost seed ranks during packing: {diagnostics}")

    return rendered_prompt, prompt_tokens, merged, diagnostics


def prepare_training_records(
    *,
    e08a_path: Path,
    records_path: Path,
    chunks_path: Path,
    documents_path: Path,
    tokenizer: Any,
    config: E45Config,
    limit: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Materialize all 5,636 canonical training records with full provenance and label masks."""
    if not e08a_path.is_file():
        raise E45Error(f"E08A results not found: {e08a_path}")
    if not records_path.is_file():
        raise E45Error(f"Records authority not found: {records_path}")
    if not chunks_path.is_file():
        raise E45Error(f"E00 chunks not found: {chunks_path}")
    if not documents_path.is_file():
        raise E45Error(f"E00 documents not found: {documents_path}")

    # Load targets
    records_raw = [json.loads(line) for line in records_path.open("r", encoding="utf-8")]
    e08a_raw = [json.loads(line) for line in e08a_path.open("r", encoding="utf-8")]

    if len(records_raw) != 5636:
        raise E45Error(f"Expected 5,636 records in {records_path}, got {len(records_raw)}")
    if len(e08a_raw) != 5636:
        raise E45Error(f"Expected 5,636 records in {e08a_path}, got {len(e08a_raw)}")

    ordered_qids = [r["question_id"] for r in records_raw]
    ordered_qids_sha = hashlib.sha256("\n".join(ordered_qids).encode("utf-8")).hexdigest()
    if ordered_qids_sha != PINNED_TRAINING_IDS_SHA256:
        raise E45Error(f"Ordered training QID hash mismatch: {ordered_qids_sha} != {PINNED_TRAINING_IDS_SHA256}")

    target_by_qid = {r["question_id"]: r for r in records_raw}

    if limit is not None:
        e08a_raw = e08a_raw[:limit]

    # Collect wanted chunks
    wanted_chunks: set[str] = set()
    for row in e08a_raw:
        for c in row["contexts"]:
            wanted_chunks.add(c["chunk_id"])

    # Scan chunks and documents
    seeds: dict[str, dict[str, Any]] = {}
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c["chunk_id"] in wanted_chunks:
                seeds[c["chunk_id"]] = c

    doc_ids = {c["document_id"] for c in seeds.values()}
    documents: dict[str, dict[str, Any]] = {}
    with documents_path.open("r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["document_id"] in doc_ids:
                documents[d["document_id"]] = d

    by_document: dict[str, list[dict[str, Any]]] = {d: [] for d in doc_ids}
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c["document_id"] in by_document:
                by_document[c["document_id"]].append(c)

    block_by_chunk: dict[str, list[dict[str, Any]]] = {}
    for doc_id, chs in by_document.items():
        for blk in parent.article_blocks(chs, documents[doc_id]):
            for c in blk:
                if c["chunk_id"] in seeds:
                    block_by_chunk[c["chunk_id"]] = blk

    context_policy = config.section("context_policy")
    materialized: list[dict[str, Any]] = []
    prompt_lengths: list[int] = []
    total_lengths: list[int] = []
    expansion_flags: list[bool] = []
    exact_e00_spans_checked = 0

    for index, row in enumerate(e08a_raw):
        qid = row["question_id"]
        target = target_by_qid[qid]
        question_text = target["question"]
        official_answer = target["answer"]
        if not official_answer or not isinstance(official_answer, str):
            raise E45Error(f"Missing or non-string official answer for row {qid}")

        # Tokenize answer and EOS
        answer_ids = tokenizer(official_answer, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
        answer_tokens_count = len(answer_ids)

        # Build seed units
        units: list[dict[str, Any]] = []
        for rank, context in enumerate(row["contexts"]):
            seed = seeds[context["chunk_id"]]
            unit = parent.seed_unit(
                seed,
                rank,
                block_by_chunk[seed["chunk_id"]],
                documents[seed["document_id"]],
                context_policy,
            )
            units.append(unit)

        # Pack evidence
        prompt_text, prompt_tokens_count, merged_spans, diagnostics = pack_training_evidence(
            question=question_text,
            units=units,
            answer_tokens_count=answer_tokens_count,
            tokenizer=tokenizer,
            config=config,
        )

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        if len(prompt_ids) != prompt_tokens_count:
            raise E45Error(f"Token count mismatch: {len(prompt_ids)} != {prompt_tokens_count}")

        input_ids = prompt_ids + answer_ids
        total_tokens_count = len(input_ids)
        if total_tokens_count > 8192:
            raise E45Error(f"Row {qid} total sequence {total_tokens_count} > 8192")

        labels = [-100] * len(prompt_ids) + answer_ids
        attention_mask = [1] * total_tokens_count

        has_expansion = bool(diagnostics.get("expansions"))
        expansion_flags.append(has_expansion)
        prompt_lengths.append(prompt_tokens_count)
        total_lengths.append(total_tokens_count)

        # Provenance verification: verify every span is exact E00 substring
        source_byte_hashes = {}
        for span in merged_spans:
            doc = documents[span["document_id"]]
            clean_text = doc["cleaned_text"]
            st, en = span["start"], span["end"]
            expected_text = clean_text[st:en]
            if span["text"] != expected_text:
                raise E45Error(f"Span text disagrees with E00 document {span['document_id']} [{st}:{en}]")
            source_byte_hashes[span["document_id"]] = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()
            exact_e00_spans_checked += 1

        record_payload = {
            "question_id": qid,
            "sample_index": index,
            "normalized_question_hash": hashlib.sha256(normalize_question(question_text).encode("utf-8")).hexdigest(),
            "ordered_seed_ids": [c["chunk_id"] for c in row["contexts"]],
            "expansion_decisions": diagnostics.get("expansions", []),
            "prompt_tokens_count": prompt_tokens_count,
            "answer_tokens_count": answer_tokens_count,
            "total_tokens_count": total_tokens_count,
            "active_label_boundaries": [len(prompt_ids), total_tokens_count],
            "rendered_prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
            "source_byte_hashes": source_byte_hashes,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # Deterministic record hash over payload without record_sha256
        rec_str = json.dumps(record_payload, sort_keys=True, ensure_ascii=False)
        record_payload["record_sha256"] = hashlib.sha256(rec_str.encode("utf-8")).hexdigest()
        materialized.append(record_payload)

        if progress_callback is not None and (index + 1) % 100 == 0:
            progress_callback(index + 1, len(e08a_raw))

    summary = {
        "records_count": len(materialized),
        "mean_prompt_length": fmean(prompt_lengths) if prompt_lengths else 0.0,
        "min_prompt_length": min(prompt_lengths) if prompt_lengths else 0,
        "max_prompt_length": max(prompt_lengths) if prompt_lengths else 0,
        "mean_total_length": fmean(total_lengths) if total_lengths else 0.0,
        "max_total_length": max(total_lengths) if total_lengths else 0,
        "expansion_retention_rate": (sum(expansion_flags) / len(expansion_flags)) if expansion_flags else 0.0,
        "answer_truncation_count": 0,
        "ordered_training_ids_sha256": ordered_qids_sha,
        "exact_e00_spans_checked": exact_e00_spans_checked,
        "canonical_prompt_renders_checked": len(materialized),
    }

    return materialized, summary


def assert_model_inventory(model: Any, config: E45Config) -> dict[str, Any]:
    """Inspect loaded model, enumerate trainable parameters, and require exact contract counts."""
    trainable_params = 0
    all_params = 0
    trainable_modules = []

    for name, param in model.named_parameters():
        num = param.numel()
        all_params += num
        if param.requires_grad:
            trainable_params += num
            trainable_modules.append(name)

    expected_trainable = config.section("model_inventory")["expected_lora_trainable_parameters"]
    if trainable_params != expected_trainable:
        raise E45Error(
            f"Observed trainable parameters {trainable_params} != expected {expected_trainable}"
        )

    generator_published = config.section("model_inventory")["generator_published_parameters"]
    embedding_published = config.section("model_inventory")["embedding_published_parameters"]
    actual_stack_total = generator_published + embedding_published + trainable_params
    exclusive_limit = config.section("model_inventory")["exclusive_competition_limit"]

    if actual_stack_total >= exclusive_limit:
        raise E45Error(
            f"Model stack total {actual_stack_total} violates exclusive competition limit {exclusive_limit}"
        )

    return {
        "trainable_parameters": trainable_params,
        "expected_trainable": expected_trainable,
        "generator_published": generator_published,
        "embedding_published": embedding_published,
        "actual_stack_total": actual_stack_total,
        "exclusive_limit": exclusive_limit,
        "trainable_modules_count": len(trainable_modules),
    }


class E45TokenizedDataset:
    """In-memory or streamed dataset for Hugging Face Trainer."""

    def __init__(self, records: list[dict[str, Any]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rec = self.records[index]
        return {
            "input_ids": rec["input_ids"],
            "attention_mask": rec["attention_mask"],
            "labels": rec["labels"],
        }


class E45CausalCollator:
    """Build the frozen batch-one causal loss without prompt-wide logits.

    Qwen2's ``logits_to_keep`` accepts the exact hidden-state positions whose
    logits are needed, while ``shift_labels`` supplies their already-shifted
    targets.  E45 masks every prompt token, so materializing vocabulary logits
    for those positions is mathematically redundant and exceeds T4 memory at
    8,192 tokens.  Selecting only answer/EOS prediction positions preserves
    the same mean cross-entropy and gradients as the full ignored-label loss.
    """

    def __init__(self, tokenizer: Any):
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        if len(features) != 1:
            raise ValueError("E45 selective causal loss requires the frozen per-device batch size of one")

        max_len = max(len(f["input_ids"]) for f in features)
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for f in features:
            cur_len = len(f["input_ids"])
            padding = max_len - cur_len
            batch_input_ids.append(f["input_ids"] + [self.pad_token_id] * padding)
            batch_attention_mask.append(f["attention_mask"] + [0] * padding)
            batch_labels.append(f["labels"] + [-100] * padding)

        labels = torch.tensor(batch_labels, dtype=torch.long)
        active_label_positions = torch.nonzero(labels[0].ne(-100), as_tuple=False).flatten()
        if active_label_positions.numel() == 0:
            raise ValueError("E45 record has no active assistant target labels")
        if int(active_label_positions[0]) == 0:
            raise ValueError("E45 causal target cannot begin at input position zero")
        expected = torch.arange(
            int(active_label_positions[0]),
            int(active_label_positions[-1]) + 1,
            dtype=torch.long,
        )
        if not torch.equal(active_label_positions, expected):
            raise ValueError("E45 assistant target labels must form one contiguous suffix")

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": labels,
            # Hidden state t predicts label t+1.  These are exactly the
            # predecessor positions of the active answer+EOS labels.
            "logits_to_keep": active_label_positions - 1,
            "shift_labels": labels[:, active_label_positions],
        }
