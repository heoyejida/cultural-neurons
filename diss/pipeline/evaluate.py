"""Continuation NLL scoring of each option as log P(option | prompt).

Records sum, byte-normalized (default for prediction), and token-normalized
log-likelihood per example.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import torch
from torch import nn

from diss.core.adapters import get_adapter
from diss.core.batching import length_bucket_sort, left_pad_batch
from diss.core.gpu import get_autocast_ctx
from diss.core.io import model_slug, output_dir
from diss.prep import load_examples

LOGGER = logging.getLogger(__name__)
DEFAULT_MAX_SEQ_LEN = 1024


def _precompute_eval(examples, tokenizer, adapter, max_seq_len: int) -> dict:
    """Tokenize every (example, option) once; option count K must be constant."""
    prepend_id = adapter.prepend_token_id(tokenizer)
    out = {}
    K = None
    for ex in examples:
        if K is None:
            K = len(ex.labels)
        elif len(ex.labels) != K:
            raise NotImplementedError(
                f"Mixed option counts not supported (got K={len(ex.labels)} after K={K})"
            )
        ctx = ex.prompt_prefix
        ctx_ids = tokenizer.encode(ctx, add_special_tokens=False)
        if prepend_id is not None:
            ctx_ids = [prepend_id] + ctx_ids
        ctx_len_base = len(ctx_ids)
        per_label = {}
        for label in ex.labels:
            text = ex.option_text_by_label[label]
            full = f"{ctx} {text}"
            enc = tokenizer.encode(full, add_special_tokens=False)
            if prepend_id is not None:
                enc = [prepend_id] + enc
            cont_len = max(1, len(enc) - ctx_len_base)
            if len(enc) > max_seq_len:
                # Raise rather than head-truncate: clipping the suffix biases
                # continuation likelihoods.
                raise ValueError(
                    f"Example {ex.example_id} option {label!r} encodes to "
                    f"{len(enc)} tokens, exceeding max_seq_len={max_seq_len}. "
                    "Increase max_seq_len, or shorten the prompt prefix."
                )
            byte_len = max(1, len(text.encode("utf-8")))
            per_label[label] = {"enc": enc, "cont_len": cont_len, "byte_len": byte_len}
        out[ex.example_id] = {"per_label": per_label}
    return out


def _eval_super_batch(
    *,
    model,
    tokenizer,
    examples,
    encs,
    batch_size: int,
    max_seq_len: int,
) -> dict:
    """Super-batched NLL forward; returns scored dict keyed by example_id."""
    pad_id = tokenizer.pad_token_id or 0
    model_max_pos = getattr(
        model.config,
        "n_positions",
        getattr(model.config, "max_position_embeddings", 1024),
    )

    def _max_len(e):
        return max(len(encs[e.example_id]["per_label"][l]["enc"]) for l in e.labels)

    batches, _ = length_bucket_sort(examples, key_fn=_max_len, batch_size=batch_size)
    scored: dict = {}

    for batch in batches:
        flat_ids: list[list[int]] = []
        flat_meta: list[tuple[str, str, int, int]] = []
        for ex in batch:
            for label in ex.labels:
                rec = encs[ex.example_id]["per_label"][label]
                flat_ids.append(rec["enc"])
                flat_meta.append(
                    (ex.example_id, label, rec["cont_len"], rec["byte_len"])
                )
        ids, mask, pos = left_pad_batch(flat_ids, pad_id=pad_id, device=model.device)
        if pos.numel() > 0:
            pos = pos.clamp(max=model_max_pos - 1)

        out = model(
            input_ids=ids, attention_mask=mask, position_ids=pos, return_dict=True
        )
        logits = out["logits"][:, :-1, :]
        target = ids[:, 1:].clone()
        target[mask[:, 1:] == 0] = -100
        per_token_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")(
            logits.transpose(1, 2),
            target,
        )

        for r, (ex_id, label, cl, bl) in enumerate(flat_meta):
            cont_loss = per_token_loss[r, -cl:].sum().item()
            d = scored.setdefault(ex_id, {"nll_sum": {}, "n_tok": {}, "n_byte": {}})
            d["nll_sum"][label] = float(cont_loss)
            d["n_tok"][label] = int(cl)
            d["n_byte"][label] = int(bl)

    return scored


def _metadata_fields(ex) -> dict:
    return {
        "topic": ex.metadata.get("_topic", ""),
        "subject": ex.metadata.get("subject", ""),
        "subject_category": ex.metadata.get("subject_category", ""),
    }


def evaluate_examples(
    *,
    model,
    tokenizer,
    model_name: str,
    examples,
    batch_size: int,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    encs: dict | None = None,
    family: str | None = None,
) -> list[dict]:
    """One result dict per example. ``encs`` (precomputed tokenization) is
    built fresh when None; pass it to avoid re-tokenizing across repeated calls.
    """
    if not examples:
        return []
    if encs is None:
        adapter = get_adapter(model_name, family=family)
        encs = _precompute_eval(examples, tokenizer, adapter, max_seq_len)

    model.eval()
    scored = _eval_super_batch(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        encs=encs,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
    )

    results: list[dict] = []
    for ex in examples:
        s = scored[ex.example_id]
        nll_sum = dict(s["nll_sum"])
        nll_byte = {l: s["nll_sum"][l] / s["n_byte"][l] for l in ex.labels}
        nll_token = {l: s["nll_sum"][l] / s["n_tok"][l] for l in ex.labels}
        pred_sum = min(nll_sum, key=nll_sum.get)
        pred_byte = min(nll_byte, key=nll_byte.get)
        pred_token = min(nll_token, key=nll_token.get)
        pred = pred_byte
        results.append(
            {
                "id": ex.example_id,
                "country": ex.country,
                "locale": ex.locale,
                "prediction": pred,
                "prediction_text": ex.option_text_by_label.get(pred, ""),
                "gold_label": ex.gold_label,
                "gold_text": ex.option_text_by_label.get(ex.gold_label, ""),
                "is_correct": pred == ex.gold_label,
                "option_nll_sum": nll_sum,
                "option_nll_byte": nll_byte,
                "option_nll_token": nll_token,
                "pred_sum": pred_sum,
                "pred_byte": pred_byte,
                "pred_token": pred_token,
                "is_correct_sum": pred_sum == ex.gold_label,
                "is_correct_byte": pred_byte == ex.gold_label,
                "is_correct_token": pred_token == ex.gold_label,
                **_metadata_fields(ex),
            }
        )
    return results


_FIELDNAMES = [
    "id",
    "country",
    "locale",
    "topic",
    "prediction",
    "prediction_text",
    "gold_label",
    "gold_text",
    "is_correct",
    "option_nll_sum",
    "option_nll_byte",
    "option_nll_token",
    "pred_sum",
    "pred_byte",
    "pred_token",
    "is_correct_sum",
    "is_correct_byte",
    "is_correct_token",
    "subject",
    "subject_category",
]

_NLL_FIELDS = ("option_nll_sum", "option_nll_byte", "option_nll_token")


def save_results(output_path: Path, results: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        w.writeheader()
        for row in results:
            out = {k: row.get(k, "") for k in _FIELDNAMES}
            for key in _NLL_FIELDS:
                if row.get(key):
                    out[key] = json.dumps(row[key])
            w.writerow(out)


def accuracy_by_attribution_split(
    results: list[dict],
    neuron_ids: set[str] | None,
) -> dict:
    """Split eval rows by whether their example_id was used in attribution.

    Mirrors CULNIG's Neuron/Non-Neuron Accuracy split: checks that mask effects
    generalize to the attribution hold-out instead of being a within-train
    artefact. Returns ``overall`` plus ``neuron``/``non_neuron`` halves when
    ``neuron_ids`` partitions the result set, each ``{"accuracy", "n"}``.
    """
    n_all = len(results)
    overall = {
        "accuracy": sum(1 for r in results if r["is_correct"]) / max(1, n_all),
        "n": n_all,
    }
    out: dict = {"overall": overall}
    if not neuron_ids:
        return out
    ids = {str(i) for i in neuron_ids}
    n_in = [r for r in results if str(r["id"]) in ids]
    n_out = [r for r in results if str(r["id"]) not in ids]
    if n_in:
        out["neuron"] = {
            "accuracy": sum(1 for r in n_in if r["is_correct"]) / len(n_in),
            "n": len(n_in),
        }
    if n_out:
        out["non_neuron"] = {
            "accuracy": sum(1 for r in n_out if r["is_correct"]) / len(n_out),
            "n": len(n_out),
        }
    return out


def _load_existing_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    filtered: list[dict] = []
    dropped_legacy = 0
    for r in rows:
        # Drop rows missing the sum/byte/token schema.
        if not (
            r.get("option_nll_sum")
            and r.get("option_nll_byte")
            and r.get("option_nll_token")
        ):
            dropped_legacy += 1
            continue
        for bool_key in (
            "is_correct",
            "is_correct_sum",
            "is_correct_byte",
            "is_correct_token",
        ):
            if bool_key in r:
                r[bool_key] = str(r.get(bool_key, "")).lower() == "true"
        for key in _NLL_FIELDS:
            if r.get(key):
                try:
                    r[key] = json.loads(r[key])
                except (json.JSONDecodeError, TypeError):
                    r[key] = None
        if any(not isinstance(r[key], dict) or not r[key] for key in _NLL_FIELDS):
            dropped_legacy += 1
            continue
        filtered.append(r)

    if dropped_legacy:
        LOGGER.warning(
            "Dropped %d legacy-schema row(s) from %s; will re-evaluate.",
            dropped_legacy,
            path,
        )
    return filtered


def _matching_cached_results(rows, examples) -> dict:
    requested = {ex.example_id: ex for ex in examples}
    if len(requested) != len(examples):
        raise ValueError("Evaluation examples must have unique IDs")
    cached = {}
    for row in rows:
        ex = requested.get(row["id"])
        if ex is not None and (
            row.get("locale") == ex.locale
            and row.get("country") == ex.country
            and row.get("gold_label") == ex.gold_label
            and row.get("gold_text") == ex.option_text_by_label.get(ex.gold_label, "")
        ):
            cached[ex.example_id] = row
    return cached


def evaluation_cache_complete(output_path: Path, **data_kwargs) -> bool:
    if not output_path.exists():
        return False
    examples = load_examples(**data_kwargs)
    rows = _load_existing_results(output_path)
    cached = _matching_cached_results(rows, examples)
    return len(rows) == len(cached) == len(examples)


def run_evaluation(
    *,
    model_name: str,
    dataset_name: str,
    target_countries: list[str] | None,
    target_locales: list[str] | None,
    data_root: Path,
    batch_size: int,
    dtype: str,
    output_subdir: str | None = None,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    target_data: str = "all",
    revision: str | None = None,
    limit_examples: int | None = None,
    model=None,
    tokenizer=None,
    devices: list[int] | None = None,
    compile_model: bool | None = None,
    compile_mode: str | None = None,
    force: bool = False,
    family: str | None = None,
) -> tuple[list[dict], Path, float]:
    slug = model_slug(model_name)
    out_dir = output_dir(slug, output_subdir or dataset_name, revision=revision)
    output_path = out_dir / "results.csv"

    if force and output_path.exists():
        LOGGER.info("--force: deleting existing %s", output_path)
        output_path.unlink()

    examples = load_examples(
        dataset_name=dataset_name,
        target_countries=target_countries,
        target_locales=target_locales,
        data_root=data_root,
        target_data=target_data,
    )

    if limit_examples is not None and len(examples) > limit_examples:
        examples = examples[:limit_examples]

    cached = _matching_cached_results(_load_existing_results(output_path), examples)
    existing = [cached[ex.example_id] for ex in examples if ex.example_id in cached]
    done_ids = {r["id"] for r in existing}
    remaining = [ex for ex in examples if ex.example_id not in done_ids]

    if not examples:
        save_results(output_path, [])
        LOGGER.warning(
            "No examples found for %s (countries=%s, locales=%s), skipping",
            dataset_name,
            target_countries,
            target_locales,
        )
        return [], output_path, 0.0

    if not remaining and existing:
        save_results(output_path, existing)
        LOGGER.info(
            "All %d examples already evaluated, skipping -> %s",
            len(examples),
            output_path,
        )
        accuracy = sum(1 for r in existing if r["is_correct"]) / max(1, len(existing))
        return existing, output_path, accuracy

    if existing:
        LOGGER.info(
            "Resuming: %d/%d done, %d remaining",
            len(done_ids),
            len(examples),
            len(remaining),
        )

    if model is None or tokenizer is None:
        from diss.core.gpu import load_model_and_tokenizer

        effective_mode = compile_mode or "reduce-overhead"
        model, tokenizer = load_model_and_tokenizer(
            model_name,
            dtype=dtype,
            devices=devices,
            compile_model=compile_model,
            compile_mode=effective_mode,
            revision=revision,
        )

    with torch.inference_mode(), get_autocast_ctx(dtype):
        new_results = evaluate_examples(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            examples=remaining,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            family=family,
        )

    by_id = {r["id"]: r for r in existing + new_results}
    results = [by_id[ex.example_id] for ex in examples]
    save_results(output_path, results)
    accuracy = sum(1 for r in results if r["is_correct"]) / max(1, len(results))
    return results, output_path, accuracy
