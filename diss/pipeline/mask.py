"""Step 4: Neuron masking — zero-out selected neurons and re-evaluate."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import torch

from diss.core.adapters import get_adapter
from diss.core.gpu import (
    get_autocast_ctx,
    load_model_and_tokenizer,
    setup_torch_optimizations,
)
from diss.core.io import (
    dump_json,
    load_json,
    masked_output_dir,
    model_slug,
    selection_records_from_json,
)
from diss.core.types import SelectionRecord
from diss.prep import load_examples
from diss.pipeline.evaluate import (
    accuracy_by_attribution_split,
    evaluate_examples,
    save_results,
    DEFAULT_MAX_SEQ_LEN,
)

LOGGER = logging.getLogger(__name__)


def _make_in_place_op(indices, op: str, device):
    """In-place zero-out closure for the selected neuron indices."""
    # Build the long-tensor once to avoid per-forward list-to-tensor overhead.
    indices_t = torch.tensor(sorted(indices), dtype=torch.long, device=device)

    if op != "suppress":
        raise ValueError(f"Unsupported operation: {op}")

    def fn(tensor):
        tensor.index_fill_(-1, indices_t, 0.0)
        return tensor

    return fn


def register_neuron_hooks(
    model,
    model_name: str,
    neurons: list[SelectionRecord],
    operation: str = "suppress",
    family: str | None = None,
):
    adapter = get_adapter(model_name, family=family)
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for r in neurons:
        grouped[(r.coord.logical_module_name, r.coord.layer_idx)].append(
            r.coord.neuron_idx
        )

    hooks = []
    for (mn, li), idxs in grouped.items():
        translated = adapter.translate_output_indices(model, li, mn, sorted(idxs))
        module = adapter.resolve_module(model, li, mn)
        kind = adapter.module_hook_kind(mn)
        device = next(module.parameters()).device
        op_fn = _make_in_place_op(translated, operation, device)

        if kind == "pre":

            def _pre_hook(_m, inputs, fn=op_fn):
                modified = fn(inputs[0])
                return (modified,) + inputs[1:]

            hooks.append(module.register_forward_pre_hook(_pre_hook))
        else:

            def _post_hook(_m, _i, output, fn=op_fn):
                return fn(output)

            hooks.append(module.register_forward_hook(_post_hook))
    return hooks


def run_masked_evaluation(
    *,
    model_name: str,
    dataset_name: str,
    neuron_file: str,
    target_countries: list[str] | None,
    target_locales: list[str] | None,
    data_root: Path,
    batch_size: int,
    dtype: str,
    operation: str = "suppress",
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    devices: list[int] | None = None,
    model=None,
    tokenizer=None,
    target_data: str = "all",
    output_subdir: str | None = None,
    revision: str | None = None,
    limit_examples: int | None = None,
    compile_model: bool | None = None,
    compile_mode: str | None = None,
    examples: list | None = None,
    encs: dict | None = None,
    family: str | None = None,
) -> tuple[list[dict], Path, float]:
    """Zero-out selected neurons and re-evaluate.

    Pre-loaded ``examples``/``encs`` may be passed in; when ``encs`` is given,
    ``evaluate_examples`` skips the precompute pass.
    """
    if model is None or tokenizer is None:
        effective_mode = compile_mode or "reduce-overhead"
        model, tokenizer = load_model_and_tokenizer(
            model_name,
            dtype=dtype,
            devices=devices,
            revision=revision,
            compile_model=compile_model,
            compile_mode=effective_mode,
        )

    if examples is None:
        examples = load_examples(
            dataset_name=dataset_name,
            target_countries=target_countries,
            target_locales=target_locales,
            data_root=data_root,
            target_data=target_data,
        )

    if limit_examples is not None and len(examples) > limit_examples:
        examples = examples[:limit_examples]

    neuron_payload = load_json(Path(neuron_file))
    neurons = selection_records_from_json(neuron_payload["top_neurons"])
    hooks = register_neuron_hooks(
        model, model_name, neurons, operation=operation, family=family
    )
    try:
        with torch.inference_mode(), get_autocast_ctx(dtype):
            results = evaluate_examples(
                model=model,
                tokenizer=tokenizer,
                model_name=model_name,
                examples=examples,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                encs=encs,
                family=family,
            )
    finally:
        for h in hooks:
            h.remove()

    slug = model_slug(model_name)
    stem = Path(neuron_file).stem
    subdir = output_subdir or dataset_name
    out_dir = masked_output_dir(slug, stem, subdir, revision=revision)
    fname = "results.csv"
    output_path = out_dir / fname
    save_results(output_path, results)

    # CULNIG attribution-split check: mask effects confined to the in-train
    # (neuron) half mean the selection is overfit rather than transferable.
    ds_ids_field = neuron_payload.get("dataset_ids", {}) or {}
    attribution_ids: set[str] = set()
    for ds_key, id_list in ds_ids_field.items():
        # Include every source dataset's ids: random/same-layer baselines copy
        # dataset_ids from the score file, so the split works across baselines.
        if isinstance(id_list, list):
            attribution_ids.update(str(i) for i in id_list)
    # Intersect with eval ids so CountryRC ids don't inflate `neuron` on BLEnD.
    eval_ids = {str(r["id"]) for r in results}
    attribution_ids &= eval_ids
    split = accuracy_by_attribution_split(results, attribution_ids)

    split_path = out_dir / "split_accuracy.json"
    dump_json(
        split_path,
        {
            "neuron_file": str(neuron_file),
            "dataset": dataset_name,
            "operation": operation,
            **split,
        },
    )

    overall_n = split["overall"]["n"]
    overall_acc = split["overall"]["accuracy"]
    if "neuron" in split and "non_neuron" in split:
        LOGGER.info(
            "Mask-eval [%s]: overall=%.4f (n=%d) | neuron-half=%.4f (n=%d) | "
            "non-neuron-half=%.4f (n=%d) | Δ(neuron−non)=%+.4f",
            stem,
            overall_acc,
            overall_n,
            split["neuron"]["accuracy"],
            split["neuron"]["n"],
            split["non_neuron"]["accuracy"],
            split["non_neuron"]["n"],
            split["neuron"]["accuracy"] - split["non_neuron"]["accuracy"],
        )
    else:
        LOGGER.info(
            "Mask-eval [%s]: overall=%.4f (n=%d) — attribution-id split "
            "unavailable (neuron file has no overlapping dataset_ids)",
            stem,
            overall_acc,
            overall_n,
        )

    return results, output_path, overall_acc


def run_mask_eval_cli(args) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    setup_torch_optimizations()
    # Default compile OFF: bf16 drift breaks Tier 1 equivalence; opt in via
    # --compile-mode, --no-compile is explicit-off.
    compile_mode_arg = getattr(args, "compile_mode", None)
    no_compile = getattr(args, "no_compile", False)
    compile_model_arg = False if (no_compile or compile_mode_arg is None) else True
    compile_mode_arg = compile_mode_arg or "reduce-overhead"
    results, path, acc = run_masked_evaluation(
        model_name=args.model,
        dataset_name=args.dataset,
        neuron_file=args.neuron_file,
        target_countries=args.target_countries,
        target_locales=args.target_locales,
        data_root=Path(args.data_root),
        batch_size=args.batch_size,
        dtype=args.dtype,
        max_seq_len=args.max_seq_len,
        devices=args.devices,
        target_data=getattr(args, "target_data", "all"),
        output_subdir=getattr(args, "output_subdir", None),
        revision=getattr(args, "revision", None),
        limit_examples=getattr(args, "limit_examples", None),
        compile_model=compile_model_arg,
        compile_mode=compile_mode_arg,
    )
    LOGGER.info("Saved %s", path)
    LOGGER.info(
        "Accuracy %.4f (%d/%d)",
        acc,
        sum(1 for r in results if r["is_correct"]),
        len(results),
    )
