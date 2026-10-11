"""Neuron attribution via activation x gradient on the continuation margin.

m(x) = ell_gold(x) - ell_distractor(x), where ell_k = -NLL_byte(option_k | prefix)
and distractor = argmax_{k != gold} ell_k. Per-neuron score = max-over-positions
of (act * grad), summed across the gold and distractor forward passes.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager, nullcontext
import gc
import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from diss.core.adapters import get_adapter
from diss.core.batching import length_bucket_sort, left_pad_batch
from diss.core.gpu import (
    get_autocast_ctx,
    load_model_and_tokenizer,
    setup_torch_optimizations,
)
from diss.core.io import dump_json, model_slug, output_dir, score_records_to_json
from diss.core.types import NeuronCoord, ScoreRecord
from diss.prep import load_examples

LOGGER = logging.getLogger(__name__)


def _forward_no_cache(model, **kwargs):
    """Forward without KV cache, falling back when the model rejects kwargs."""
    try:
        return model(**kwargs, use_cache=False)
    except TypeError as exc:
        if "logits_to_keep" in kwargs and "logits_to_keep" in str(exc):
            kwargs.pop("logits_to_keep")
            return _forward_no_cache(model, **kwargs)
        if "use_cache" not in str(exc):
            raise
        return model(**kwargs)


@contextmanager
def _disable_parameter_grads(model):
    """Temporarily freeze parameters so only activation grads are retained."""
    params = list(model.parameters())
    states = [p.requires_grad for p in params]
    try:
        for p in params:
            p.requires_grad_(False)
            p.grad = None
        yield
    finally:
        for p, requires_grad in zip(params, states):
            p.requires_grad_(requires_grad)


def _is_gradient_checkpointing_enabled(model) -> bool:
    state = getattr(model, "is_gradient_checkpointing", False)
    return bool(state() if callable(state) else state)


@contextmanager
def _checkpoint_grad_scoring(model):
    """Force grad checkpointing and KV cache off for the attribution forward.

    Checkpointing is incompatible with retain_grad captures: use_reentrant=False
    disconnects captured leaves from the backward path (grads all None);
    use_reentrant=True OOMs faster than save_on_cpu offload alone.
    """
    config = getattr(model, "config", None)
    old_use_cache = getattr(config, "use_cache", None)
    was_checkpointing = _is_gradient_checkpointing_enabled(model)
    disabled_checkpointing = False
    if was_checkpointing:
        disable = getattr(model, "gradient_checkpointing_disable", None)
        if disable is not None:
            disable()
            disabled_checkpointing = True

    if config is not None and old_use_cache is not None:
        config.use_cache = False
    try:
        yield
    finally:
        if config is not None and old_use_cache is not None:
            config.use_cache = old_use_cache
        if disabled_checkpointing:
            enable = getattr(model, "gradient_checkpointing_enable", None)
            if enable is not None:
                try:
                    enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                except TypeError:
                    enable()


def _saved_tensors_cpu_ctx():
    save_on_cpu = getattr(torch.autograd.graph, "save_on_cpu", None)
    if save_on_cpu is None or not torch.cuda.is_available():
        return nullcontext()
    try:
        return save_on_cpu(pin_memory=True)
    except TypeError:
        return save_on_cpu()


def _register_capture_hooks(model, adapter):
    """Register per-module capture hooks; returns (hooks, captures).

    captures is keyed by (logical_module_name, layer_idx) and holds the
    retain_grad-tagged activation. Hooks are no-ops during inference so the
    same registration is safe across the pre-pass and the grad-enabled passes.
    """
    captures: dict[tuple[str, int], torch.Tensor] = {}
    hooks = []

    def _save_post(mod_name: str, layer: int):
        def hook(_m, _i, output):
            if torch.is_inference_mode_enabled() or not torch.is_grad_enabled():
                return None
            if not output.requires_grad:
                return None
            captures[(mod_name, layer)] = output
            output.retain_grad()

        return hook

    def _save_pre(mod_name: str, layer: int):
        def hook(_m, inputs):
            if torch.is_inference_mode_enabled() or not torch.is_grad_enabled():
                return None
            tensor = inputs[0]
            if not tensor.requires_grad:
                return None
            captures[(mod_name, layer)] = tensor
            tensor.retain_grad()

        return hook

    for li in range(adapter.get_num_layers(model)):
        for mn in adapter.default_score_modules():
            kind = adapter.module_hook_kind(mn)
            module = adapter.resolve_module(model, li, mn)
            if kind == "pre":
                hooks.append(module.register_forward_pre_hook(_save_pre(mn, li)))
            else:
                hooks.append(module.register_forward_hook(_save_post(mn, li)))
    return hooks, captures


def _allocate_accumulator(model, adapter, num_countries: int) -> dict:
    accum: dict[tuple[str, int], torch.Tensor] = {}
    num_layers = adapter.get_num_layers(model)
    for li in range(num_layers):
        for mn in adapter.default_score_modules():
            module = adapter.resolve_module(model, li, mn)
            H = adapter.logical_module_size(model, li, mn)
            device = next(module.parameters()).device
            accum[(mn, li)] = torch.zeros(
                (num_countries, H),
                dtype=torch.float32,
                device=device,
            )
    return accum


def _finalize_accumulator(accum, adapter, idx_to_country: list) -> list:
    """Move accumulator to CPU and emit ScoreRecord list, dropping all-zero columns.

    Bulk-converts surviving columns in one .tolist(); a per-cell .item() loop
    triggers millions of CUDA-host syncs on 7B-class accumulators.
    """
    records: list[ScoreRecord] = []
    for (mn, li), tensor in accum.items():
        cpu = tensor.detach().cpu()
        nonzero_cols = (cpu != 0.0).any(dim=0).nonzero(as_tuple=True)[0].tolist()
        if not nonzero_cols:
            continue
        cols = cpu[:, nonzero_cols].t().tolist()
        for k, ni in enumerate(nonzero_cols):
            scores = {idx_to_country[ci]: v for ci, v in enumerate(cols[k]) if v != 0.0}
            if not scores:
                continue
            coord = NeuronCoord(adapter.family, mn, li, int(ni))
            records.append(ScoreRecord(coord, scores))
    return records


def _precompute_encodings(
    examples,
    tokenizer,
    adapter,
    max_seq_len: int,
) -> dict:
    """Tokenize every (example, option) once, keyed by example_id."""
    prepend_id = adapter.prepend_token_id(tokenizer)
    out = {}
    for e in examples:
        ctx = e.prompt_prefix or ""
        ctx_ids = tokenizer.encode(ctx, add_special_tokens=False)
        if prepend_id is not None:
            ctx_ids = [prepend_id] + ctx_ids
        ctx_len = len(ctx_ids)
        per_label = {}
        for label in e.labels:
            text = e.option_text_by_label[label]
            full = f"{ctx} {text}"
            enc = tokenizer.encode(full, add_special_tokens=False)
            if prepend_id is not None:
                enc = [prepend_id] + enc
            cont_len = max(1, len(enc) - ctx_len)
            if len(enc) > max_seq_len:
                raise ValueError(
                    f"Example {e.example_id} option {label!r} encodes to "
                    f"{len(enc)} tokens, exceeding max_seq_len={max_seq_len}. "
                    "Increase max_seq_len, or shorten the prompt prefix."
                )
            byte_len = max(1, len(text.encode("utf-8")))
            per_label[label] = {"enc": enc, "cont_len": cont_len, "byte_len": byte_len}
        out[e.example_id] = {"ctx_len": ctx_len, "per_label": per_label}
    return out


def _continuation_ells_from_logits(
    logits: torch.Tensor,
    ids: torch.Tensor,
    meta: list[tuple[int, int]],
) -> torch.Tensor:
    """Return byte-normalized continuation log-likelihood per row.

    Avoids materializing log_softmax over all positions; only continuation
    targets enter the loss, the memory-critical path for 7B attribution.
    """
    has_full_sequence_logits = logits.shape[1] == ids.shape[1]
    if has_full_sequence_logits:
        shift_logits = logits[:, :-1, :]
        shift_targets = ids[:, 1:]
    ells = []
    for row, (cont_len, byte_len) in enumerate(meta):
        if has_full_sequence_logits:
            row_logits = shift_logits[row, -cont_len:, :]
            row_targets = shift_targets[row, -cont_len:]
        else:
            if logits.shape[1] < cont_len + 1:
                raise ValueError(
                    f"Need at least {cont_len + 1} logits positions for continuation scoring; "
                    f"got {logits.shape[1]}"
                )
            row_logits = logits[row, -(cont_len + 1) : -1, :]
            row_targets = ids[row, -cont_len:]
        nll = F.cross_entropy(row_logits, row_targets, reduction="sum")
        ells.append(-nll / byte_len)
    return torch.stack(ells)


def _select_strongest_distractors(
    *,
    model,
    examples,
    encs,
    batch_size: int,
    autocast,
    pad_id: int,
) -> dict[str, str]:
    """Batched inference over (B*K, S); returns distractor label per example_id."""
    distractor: dict[str, str] = {}

    def _max_len(e):
        return max(len(encs[e.example_id]["per_label"][l]["enc"]) for l in e.labels)

    batches, _ = length_bucket_sort(examples, key_fn=_max_len, batch_size=batch_size)
    n_batches = len(batches)
    log_every = max(1, n_batches // 10)
    for bi, batch in enumerate(batches):
        if bi % log_every == 0:
            LOGGER.info("distractor pre-pass: batch %d/%d", bi + 1, n_batches)
        flat_ids: list[list[int]] = []
        flat_meta: list[tuple[str, str, int, int]] = (
            []
        )  # (ex_id, label, cont_len, byte_len)
        for e in batch:
            for label in e.labels:
                rec = encs[e.example_id]["per_label"][label]
                flat_ids.append(rec["enc"])
                flat_meta.append(
                    (e.example_id, label, rec["cont_len"], rec["byte_len"])
                )
        ids, mask, pos = left_pad_batch(flat_ids, pad_id=pad_id, device=model.device)
        logits_to_keep = max(cl for _, _, cl, _ in flat_meta) + 1
        with torch.inference_mode(), autocast:
            out = _forward_no_cache(
                model,
                input_ids=ids,
                attention_mask=mask,
                position_ids=pos,
                return_dict=True,
                logits_to_keep=logits_to_keep,
            )
            ells = _continuation_ells_from_logits(
                out.logits,
                ids,
                [(cl, bl) for _, _, cl, bl in flat_meta],
            )

        ells_per_ex: dict[str, dict[str, float]] = defaultdict(dict)
        for ell, (ex_id, label, _cl, _bl) in zip(
            ells.detach().cpu().tolist(), flat_meta
        ):
            ells_per_ex[ex_id][label] = float(ell)
        for e in batch:
            gold = e.gold_label
            others = [l for l in e.labels if l != gold]
            ells = ells_per_ex[e.example_id]
            distractor[e.example_id] = max(others, key=lambda l: ells[l])
    return distractor


def _score_gold_vs_distractor_margin(
    *,
    model,
    adapter,
    examples,
    encs,
    distractor: dict[str, str],
    batch_size: int,
    autocast,
    pad_id: int,
    accum: dict,
    country_to_idx: dict[str, int],
    captures: dict,
    empty_cache_interval: int = 0,
) -> dict[str, float]:
    """Score gold (sign +1) and distractor (sign -1) rows in one (2B, S) pass.

    empty_cache_interval=0 leaves steady-state reuse to the CUDA allocator;
    N>0 forces cleanup every N batches (only needed when monitoring shows
    cumulative growth, e.g. 7B with save_on_cpu offload).
    """
    margin_totals: dict[str, float] = defaultdict(float)

    def _row_len(e):
        gold_l = len(encs[e.example_id]["per_label"][e.gold_label]["enc"])
        dist_l = len(encs[e.example_id]["per_label"][distractor[e.example_id]]["enc"])
        return max(gold_l, dist_l)

    batches, _ = length_bucket_sort(examples, key_fn=_row_len, batch_size=batch_size)
    n_batches = len(batches)
    log_every = max(1, n_batches // 10)

    for bi, batch in enumerate(batches):
        if bi % log_every == 0:
            LOGGER.info("grad-scoring: batch %d/%d", bi + 1, n_batches)

        row_ids, row_meta, signs = [], [], []
        for e in batch:
            gold_rec = encs[e.example_id]["per_label"][e.gold_label]
            dist_rec = encs[e.example_id]["per_label"][distractor[e.example_id]]
            row_ids.append(gold_rec["enc"])
            row_meta.append((gold_rec["cont_len"], gold_rec["byte_len"]))
            signs.append(+1.0)
            row_ids.append(dist_rec["enc"])
            row_meta.append((dist_rec["cont_len"], dist_rec["byte_len"]))
            signs.append(-1.0)

        ids, mask, pos = left_pad_batch(row_ids, pad_id=pad_id, device=model.device)
        logits_to_keep = max(cl for cl, _ in row_meta) + 1
        signs_t = torch.tensor(signs, dtype=torch.float32, device=model.device)
        country_idx_2b = torch.tensor(
            [country_to_idx[e.country] for e in batch for _ in range(2)],
            dtype=torch.long,
            device=model.device,
        )

        captures.clear()
        model.zero_grad(set_to_none=True)
        with _checkpoint_grad_scoring(model), _saved_tensors_cpu_ctx():
            with autocast:
                out = _forward_no_cache(
                    model,
                    input_ids=ids,
                    attention_mask=mask,
                    position_ids=pos,
                    return_dict=True,
                    logits_to_keep=logits_to_keep,
                )
                ells = _continuation_ells_from_logits(out.logits, ids, row_meta)
            loss = (signs_t * ells).sum()
            loss.backward(retain_graph=False)

        if bi == 0:
            n_caps = len(captures)
            n_with_grad = sum(1 for t in captures.values() if t.grad is not None)
            LOGGER.info(
                "grad-scoring sanity: captures=%d, with_grad=%d (after first batch)",
                n_caps,
                n_with_grad,
            )
            if n_caps == 0:
                raise RuntimeError(
                    "Attribution: hooks captured 0 tensors after first backward. "
                    "Check that gradient checkpointing is using use_reentrant=False."
                )
            if n_with_grad == 0:
                raise RuntimeError(
                    f"Attribution: {n_caps} captures but ALL .grad are None. "
                    "Likely use_reentrant=True is in effect — the first forward "
                    "ran under no_grad and hooks did not enter the autograd graph."
                )

        with torch.no_grad():
            for (mn, li), tensor in captures.items():
                grads = tensor.grad
                if grads is None:
                    continue
                logical_tensor = adapter.extract_logical_tensor(model, li, mn, tensor)
                logical_grads = adapter.extract_logical_tensor(model, li, mn, grads)
                scores = logical_tensor * logical_grads
                neg_inf = torch.finfo(scores.dtype).min
                scores = scores.masked_fill(
                    mask.to(scores.device).unsqueeze(-1) == 0, neg_inf
                )
                mx = scores.max(dim=1).values
                mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
                accum[(mn, li)].index_add_(
                    0, country_idx_2b.to(mx.device), mx.float()
                )

        if bi == 0 and any(t.requires_grad for t in accum.values()):
            raise RuntimeError(
                "Attribution: accumulator became grad-tracking — the scatter "
                "must run under torch.no_grad() (per-batch autograd-graph leak)."
            )

        ells_cpu = ells.detach().cpu()
        captures.clear()
        for r, e in enumerate(batch):
            margin_totals[e.country] += float(ells_cpu[2 * r] - ells_cpu[2 * r + 1])

        del out, ells, loss, ids, mask, pos, signs_t, country_idx_2b
        if empty_cache_interval > 0 and (bi + 1) % empty_cache_interval == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return margin_totals


def calculate_scores(
    model,
    tokenizer,
    model_name: str,
    examples,
    batch_size: int,
    dtype: str = "bfloat16",
    max_seq_len: int = 1024,
    empty_cache_interval: int = 0,
    family: str | None = None,
) -> tuple[list[ScoreRecord], dict[str, float], list[str]]:
    adapter = get_adapter(model_name, family=family)
    countries_sorted = sorted({e.country for e in examples})
    country_to_idx = {c: i for i, c in enumerate(countries_sorted)}

    hooks, captures = _register_capture_hooks(model, adapter)
    adapter.prepare_for_scoring(model, max_seq_len)
    autocast = get_autocast_ctx(dtype)
    pad_id = tokenizer.pad_token_id or 0
    was_training = model.training
    LOGGER.info(
        "Attribution continuation scoring: examples=%d batch_size=%d max_seq_len=%d "
        "grad_checkpointing=off save_on_cpu=%s logits_to_keep=on",
        len(examples),
        batch_size,
        max_seq_len,
        bool(getattr(torch.autograd.graph, "save_on_cpu", None))
        and torch.cuda.is_available(),
    )

    try:
        accum = _allocate_accumulator(
            model, adapter, num_countries=len(countries_sorted)
        )

        encs = _precompute_encodings(examples, tokenizer, adapter, max_seq_len)

        with _disable_parameter_grads(model):
            distractor = _select_strongest_distractors(
                model=model,
                examples=examples,
                encs=encs,
                batch_size=batch_size,
                autocast=autocast,
                pad_id=pad_id,
            )

        model.eval()
        with torch.enable_grad():
            margin_totals = _score_gold_vs_distractor_margin(
                model=model,
                adapter=adapter,
                examples=examples,
                encs=encs,
                distractor=distractor,
                batch_size=batch_size,
                autocast=autocast,
                pad_id=pad_id,
                accum=accum,
                country_to_idx=country_to_idx,
                captures=captures,
                empty_cache_interval=empty_cache_interval,
            )
    finally:
        model.train(was_training)
        model.zero_grad(set_to_none=True)
        for h in hooks:
            h.remove()

    records = _finalize_accumulator(accum, adapter, countries_sorted)
    dataset_ids = sorted({e.example_id for e in examples})
    return records, dict(margin_totals), dataset_ids


def run_scoring_cli(args, *, model=None, tokenizer=None) -> None:
    """CLI entry for `python -m diss.cli score`.

    Pass a preloaded model and tokenizer to skip load_model_and_tokenizer;
    the run-attribution orchestrator uses this to amortize one model load.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    setup_torch_optimizations()
    revision = getattr(args, "revision", None)
    target_locales = getattr(args, "target_locales", None)
    if "countryrc_localized" in args.datasets:
        from diss.core.locale_meta import LOCALE_TO_COUNTRY

        if not target_locales or any(loc not in LOCALE_TO_COUNTRY for loc in target_locales):
            raise ValueError("CountryRC scoring requires valid --target-locales")
    if model is None or tokenizer is None:
        compile_mode_arg = getattr(args, "compile_mode", None)
        no_compile = getattr(args, "no_compile", False)
        compile_model = False if (no_compile or compile_mode_arg is None) else True
        compile_mode = compile_mode_arg or "default"
        model, tokenizer = load_model_and_tokenizer(
            args.model,
            dtype=args.dtype,
            devices=args.devices,
            revision=revision,
            compile_model=compile_model,
            compile_mode=compile_mode,
        )
    slug = model_slug(args.model)
    jobs = [
        (ds, [loc] if loc is not None else target_locales)
        for ds in args.datasets
        for loc in (dict.fromkeys(target_locales) if ds == "countryrc_localized" else [None])
    ]
    for ds, ds_target_locales in jobs:
        LOGGER.info(
            "Scoring %s for %s (revision=%s)", ds, args.model, revision or "default"
        )
        if ds == "countryrc_localized":
            ds_target_countries = None
        else:
            ds_target_countries = args.target_countries
        examples = load_examples(
            dataset_name=ds,
            target_countries=ds_target_countries,
            target_locales=ds_target_locales,
            data_root=Path(args.data_root),
            target_data=args.target_data,
        )
        limit = getattr(args, "limit_examples", None)
        if limit is not None and len(examples) > limit:
            examples = examples[:limit]
        records, totals, ids = calculate_scores(
            model,
            tokenizer,
            args.model,
            examples,
            args.batch_size,
            args.dtype,
            max_seq_len=getattr(args, "max_seq_len", 1024),
            empty_cache_interval=getattr(args, "empty_cache_interval", 0),
            family=getattr(args, "family", None),
        )

        layer_summary: dict[int, dict] = defaultdict(lambda: {"count": 0, "sum": 0.0})
        for r in records:
            layer = r.coord.layer_idx
            total_score = sum(r.scores.values())
            layer_summary[layer]["count"] += 1
            layer_summary[layer]["sum"] += total_score
        layer_summary_out = {
            str(li): {
                "count": v["count"],
                "sum": round(v["sum"], 6),
                "mean": round(v["sum"] / v["count"], 6) if v["count"] else 0.0,
            }
            for li, v in sorted(layer_summary.items())
        }

        score_name = f"{ds}_{ds_target_locales[0]}" if ds == "countryrc_localized" else ds
        out = (
            output_dir(slug, "cultural_neuron", revision=revision)
            / f"{score_name}_max_scores.json"
        )
        dump_json(
            out,
            {
                "model_name": args.model,
                "dataset_name": ds,
                "revision": revision,
                "normalization": "byte",
                "margin": "gold_minus_strongest_distractor",
                "neuron_scores": score_records_to_json(records),
                "total_probabilities_per_country": totals,
                "dataset_ids": {ds: ids},
                "layer_summary": layer_summary_out,
            },
        )
        LOGGER.info("Saved %s", out)
