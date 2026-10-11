"""Anchor-based checkpoint-trace orchestrator.

The final checkpoint's cultural-neuron selection is the fixed anchor; earlier
checkpoints re-run only the masking against that same neuron set, so per-cp
masking drops are not confounded by selection drift. Config:
``diss/config/checkpoint_trace.yaml``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
import yaml

from diss.core.gpu import load_model_and_tokenizer, setup_torch_optimizations
from diss.core.io import (
    OUTPUT_ROOT,
    dump_json,
    load_json,
    masked_output_dir,
    model_slug,
    output_dir,
    score_records_to_json,
)
from diss.core.registry import CULTURE_COUNTRY_BY_LOCALE
from diss.pipeline.attribute import calculate_scores
from diss.pipeline.evaluate import evaluation_cache_complete, run_evaluation
from diss.pipeline.mask import run_masked_evaluation
from diss.pipeline.select import (
    DEFAULT_MLP_THRESHOLD,
    select_culture_neurons,
    select_random_neurons,
    select_same_layer_random_neurons,
)

LOGGER = logging.getLogger(__name__)


def _read_total_selected(culture_file: Path) -> int:
    try:
        return int(load_json(culture_file).get("total_selected", 0))
    except (OSError, ValueError):
        return 0


VALID_CROSS_LOCALE_POLICIES = ("all_cps", "anchor_only")


def should_skip_cross_locale_cell(
    *,
    policy: str,
    is_anchor_cp: bool,
    src_loc: str,
    eval_loc: str,
) -> bool:
    """Skip an off-diagonal cross-locale cell only under ``anchor_only`` at a
    non-anchor cp. Single source of truth for this decision; unit-tested."""
    if policy not in VALID_CROSS_LOCALE_POLICIES:
        raise ValueError(
            f"cross_locale_policy must be one of {VALID_CROSS_LOCALE_POLICIES}, "
            f"got {policy!r}"
        )
    if policy == "anchor_only" and not is_anchor_cp and src_loc != eval_loc:
        return True
    return False


@dataclass
class CheckpointJob:
    label: str
    load_kwargs: dict


def iter_checkpoints(run: dict) -> Iterator[CheckpointJob]:
    src = run["source"]
    kind = src["kind"]
    if kind == "hf_revisions":
        repo = src["repo"]
        for rev in src["revisions"]:
            yield CheckpointJob(
                label=str(rev), load_kwargs={"model_name": repo, "revision": str(rev)}
            )
    elif kind == "local_checkpoints":
        base = Path(src["base_path"])
        for cp in src["checkpoints"]:
            label = cp["label"]
            path = base / cp["path"]
            yield CheckpointJob(label=label, load_kwargs={"model_name": str(path)})
    else:
        raise ValueError(f"Unknown checkpoint source kind: {kind!r}")


def _compute_and_save_attribution(
    *,
    model,
    tokenizer,
    model_name: str,
    dataset_name: str,
    target_countries: list[str] | None,
    target_locales: list[str] | None = None,
    data_root: Path,
    score_batch: int,
    dtype: str,
    max_seq_len: int,
    empty_cache_interval: int,
    revision_label: str,
    out_path: Path,
    family: str | None = None,
) -> None:
    from diss.prep import load_examples

    examples = load_examples(
        dataset_name=dataset_name,
        target_countries=target_countries,
        target_locales=target_locales,
        data_root=data_root,
        target_data="neuron",
    )
    records, totals, ids = calculate_scores(
        model,
        tokenizer,
        model_name,
        examples,
        score_batch,
        dtype,
        max_seq_len=max_seq_len,
        empty_cache_interval=empty_cache_interval,
        family=family,
    )
    layer_summary: dict = defaultdict(lambda: {"count": 0, "sum": 0.0})
    for r in records:
        tot = sum(r.scores.values())
        layer_summary[r.coord.layer_idx]["count"] += 1
        layer_summary[r.coord.layer_idx]["sum"] += tot
    layer_summary_out = {
        str(li): {
            "count": v["count"],
            "sum": round(v["sum"], 6),
            "mean": round(v["sum"] / v["count"], 6) if v["count"] else 0.0,
        }
        for li, v in sorted(layer_summary.items())
    }
    dump_json(
        out_path,
        {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "revision": revision_label,
            "normalization": "byte",
            "margin": "gold_minus_strongest_distractor",
            "neuron_scores": score_records_to_json(records),
            "total_probabilities_per_country": totals,
            "dataset_ids": {dataset_name: ids},
            "layer_summary": layer_summary_out,
        },
    )
    LOGGER.info("Saved %s", out_path)


def _run_neuron_pipeline_for_checkpoint(
    *,
    model,
    tokenizer,
    model_name: str,
    architecture: str,
    revision_label: str,
    locales: list[str],
    defaults: dict,
    force: bool,
    anchor_paths: dict | None = None,
) -> dict | None:
    """Run the cultural-neuron pipeline for one loaded checkpoint.

    ``anchor_paths is None`` marks the anchor cp: run eval, score, select,
    baselines, and masking, then return ``{locale: {country, culture, random,
    slr}}`` (random/slr are ``{seed: Path}``; empty selection -> None).
    Otherwise run eval+score only and re-use the anchor neuron files; return None.
    """
    data_root = Path(defaults["data_root"])
    batch_size = defaults["batch_size"]
    score_batch = defaults.get("score_batch_size", batch_size)
    score_empty_cache_interval = int(defaults.get("score_empty_cache_interval", 0))
    dtype = defaults["dtype"]
    max_seq_len = defaults["max_seq_len"]
    mlp_threshold = defaults.get("mlp_threshold", DEFAULT_MLP_THRESHOLD)
    countryrc_threshold = defaults["countryrc_threshold"]
    random_seeds = defaults["random_seeds"]
    cross_locale_policy = defaults.get("cross_locale_policy", "all_cps")
    if cross_locale_policy not in VALID_CROSS_LOCALE_POLICIES:
        raise ValueError(
            f"cross_locale_policy must be one of {VALID_CROSS_LOCALE_POLICIES}, "
            f"got {cross_locale_policy!r}"
        )
    slug = model_slug(model_name)
    is_anchor_cp = anchor_paths is None

    countries: list[tuple[str, str]] = []
    for loc in locales:
        country = CULTURE_COUNTRY_BY_LOCALE.get(loc)
        if country is None:
            raise ValueError(f"No country mapping for locale {loc!r}")
        countries.append((loc, country))

    # Stage 1 — behaviour eval
    for loc, country in countries:
        for tdata, sub in [("all", f"blend_mcq_{loc}")]:
            out_path = output_dir(slug, sub, revision=revision_label) / "results.csv"
            run_evaluation(
                model_name=model_name,
                dataset_name="blend_mcq",
                target_countries=[country],
                target_locales=None,
                data_root=data_root,
                batch_size=batch_size,
                dtype=dtype,
                output_subdir=sub,
                max_seq_len=max_seq_len,
                target_data=tdata,
                revision=revision_label,
                model=model,
                tokenizer=tokenizer,
                force=force,
                family=architecture,
            )

        out_path = (
            output_dir(slug, f"belebele_{loc}", revision=revision_label) / "results.csv"
        )
        run_evaluation(
            model_name=model_name,
            dataset_name="belebele",
            target_countries=None,
            target_locales=[loc],
            data_root=data_root,
            batch_size=batch_size,
            dtype=dtype,
            output_subdir=f"belebele_{loc}",
            max_seq_len=max_seq_len,
            target_data="all",
            revision=revision_label,
            model=model,
            tokenizer=tokenizer,
            force=force,
            family=architecture,
        )

    # Stage 2 — attribution (needed at every cp for anchor tracking)
    cn_dir = output_dir(slug, "cultural_neuron", revision=revision_label)
    blend_scores_path = cn_dir / "blend_mcq_max_scores.json"
    target_country_list = [c for _, c in countries]

    if force or not blend_scores_path.exists():
        _compute_and_save_attribution(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            dataset_name="blend_mcq",
            target_countries=target_country_list,
            data_root=data_root,
            score_batch=score_batch,
            dtype=dtype,
            max_seq_len=max_seq_len,
            empty_cache_interval=score_empty_cache_interval,
            revision_label=revision_label,
            out_path=blend_scores_path,
            family=architecture,
        )
    else:
        LOGGER.info("Skip score (exists): %s", blend_scores_path)

    # CRC is locale-pinned (gold = LOCALE_TO_COUNTRY[locale]); score per locale
    # into its own file so each country's CRC filter uses only matching stimuli.
    for loc, _ in countries:
        crc_scores_path = cn_dir / f"countryrc_localized_{loc}_max_scores.json"
        if force or not crc_scores_path.exists():
            _compute_and_save_attribution(
                model=model,
                tokenizer=tokenizer,
                model_name=model_name,
                dataset_name="countryrc_localized",
                target_countries=None,
                target_locales=[loc],
                data_root=data_root,
                score_batch=score_batch,
                dtype=dtype,
                max_seq_len=max_seq_len,
                empty_cache_interval=score_empty_cache_interval,
                revision_label=revision_label,
                out_path=crc_scores_path,
                family=architecture,
            )
        else:
            LOGGER.info("Skip score (exists): %s", crc_scores_path)

    # Stage 3 — resolve per-locale anchor neuron files
    locale_anchor: list[tuple[str, str, Path | None]] = []  # (loc, country, culture)
    if is_anchor_cp:
        for loc, country in countries:
            country_tag = country.replace(" ", "")
            culture_file = cn_dir / f"{country_tag}_neurons_blend_mcq.json"
            if force or not culture_file.exists():
                try:
                    payload = select_culture_neurons(
                        model_name=model_name,
                        dataset_name="blend_mcq",
                        target_country=country,
                        mlp_threshold=mlp_threshold,
                        countryrc_threshold=countryrc_threshold,
                        revision=revision_label,
                        family=architecture,
                    )
                except ValueError as exc:
                    LOGGER.warning(
                        "select_culture_neurons (%s): %s — writing empty anchor",
                        country,
                        exc,
                    )
                    payload = {
                        "model_name": model_name,
                        "target_country": country,
                        "revision": revision_label,
                        "dataset_name": "blend_mcq",
                        "top_neurons": [],
                        "total_selected": 0,
                    }
                dump_json(culture_file, payload)
                LOGGER.info(
                    "Saved %s (selected=%d)", culture_file, payload["total_selected"]
                )

            n_selected = _read_total_selected(culture_file)
            if n_selected == 0:
                LOGGER.warning(
                    "[%s/%s] 0 culture neurons at anchor cp — masking will be skipped",
                    revision_label,
                    country,
                )
                locale_anchor.append((loc, country, None))
                continue
            locale_anchor.append((loc, country, culture_file))

            # random + same-layer-random baselines, sized to the anchor culture file
            for seed in random_seeds:
                rand_file = cn_dir / f"{country_tag}_all_neurons_random{seed}.json"
                if force or not rand_file.exists():
                    rpayload = select_random_neurons(
                        model_name=model_name,
                        dataset_name="blend_mcq",
                        seed=seed,
                        match_neuron_file=str(culture_file),
                        revision=revision_label,
                        family=architecture,
                    )
                    dump_json(rand_file, rpayload)
                    LOGGER.info("Saved %s", rand_file)
                slr_file = cn_dir / f"{country_tag}_all_neurons_random_layer{seed}.json"
                if force or not slr_file.exists():
                    slr_payload = select_same_layer_random_neurons(
                        model_name=model_name,
                        dataset_name="blend_mcq",
                        seed=seed,
                        match_neuron_file=str(culture_file),
                        revision=revision_label,
                        family=architecture,
                    )
                    dump_json(slr_file, slr_payload)
                    LOGGER.info("Saved %s", slr_file)
    else:
        for loc, country in countries:
            entry = anchor_paths.get(loc) if anchor_paths else None
            culture = (
                Path(entry["culture"])
                if entry is not None and entry.get("culture") is not None
                else None
            )
            locale_anchor.append((loc, country, culture))

    # Stage 4 — masking: culture cross-locale matrix, random/SLR diagonal only
    from diss.prep import load_examples as _load_examples
    from diss.pipeline.evaluate import _precompute_eval as _precompute_eval_fn
    from diss.core.adapters import get_adapter as _get_adapter

    _adapter = _get_adapter(model_name, family=architecture)
    _slice_cache: dict[tuple, tuple[list, dict]] = {}

    def _slice_key(ds_name, target_countries, target_locales, target_data):
        return (
            ds_name,
            tuple(target_countries) if target_countries else None,
            tuple(target_locales) if target_locales else None,
            target_data,
        )

    def _get_slice(ds_name, target_countries, target_locales, target_data):
        key = _slice_key(ds_name, target_countries, target_locales, target_data)
        cached = _slice_cache.get(key)
        if cached is not None:
            return cached
        examples = _load_examples(
            dataset_name=ds_name,
            target_countries=target_countries,
            target_locales=target_locales,
            data_root=data_root,
            target_data=target_data,
        )
        encs = _precompute_eval_fn(examples, tokenizer, _adapter, max_seq_len)
        _slice_cache[key] = (examples, encs)
        return examples, encs

    def _mask_for_neuron_file(
        neuron_file: Path, eval_loc: str, target_country: str
    ) -> None:
        for ds_name, target_args, sub_name, target_data in [
            (
                "blend_mcq",
                {"target_countries": [target_country], "target_locales": None},
                f"blend_mcq_{eval_loc}_non_neuron",
                "non_neuron",
            ),
            (
                "belebele",
                {"target_countries": None, "target_locales": [eval_loc]},
                f"belebele_{eval_loc}",
                "all",
            ),
        ]:
            masked_path = (
                masked_output_dir(
                    slug, neuron_file.stem, sub_name, revision=revision_label
                )
                / "results.csv"
            )
            if not force and evaluation_cache_complete(
                masked_path, dataset_name=ds_name, **target_args,
                data_root=data_root, target_data=target_data,
            ):
                LOGGER.info("Skip mask (exists): %s", masked_path)
                continue
            slc_examples, slc_encs = _get_slice(
                ds_name,
                target_args["target_countries"],
                target_args["target_locales"],
                target_data,
            )
            run_masked_evaluation(
                model_name=model_name,
                dataset_name=ds_name,
                neuron_file=str(neuron_file),
                target_countries=target_args["target_countries"],
                target_locales=target_args["target_locales"],
                data_root=data_root,
                batch_size=batch_size,
                dtype=dtype,
                operation="suppress",
                max_seq_len=max_seq_len,
                devices=None,
                model=model,
                tokenizer=tokenizer,
                target_data=target_data,
                output_subdir=sub_name,
                revision=revision_label,
                examples=slc_examples,
                encs=slc_encs,
                family=architecture,
            )

    # 4a. cross-locale matrix: each locale's anchor culture file on every locale's data
    for src_loc, src_country, culture_file in locale_anchor:
        if culture_file is None:
            continue
        for eval_loc in locales:
            if should_skip_cross_locale_cell(
                policy=cross_locale_policy,
                is_anchor_cp=is_anchor_cp,
                src_loc=src_loc,
                eval_loc=eval_loc,
            ):
                continue
            target_country = CULTURE_COUNTRY_BY_LOCALE[eval_loc]
            _mask_for_neuron_file(culture_file, eval_loc, target_country)

    # 4b. random + SLR diagonal
    for loc, country, culture_file in locale_anchor:
        if culture_file is None:
            continue
        country_tag = country.replace(" ", "")
        if is_anchor_cp:
            rand_files = [
                cn_dir / f"{country_tag}_all_neurons_random{s}.json"
                for s in random_seeds
            ]
            slr_files = [
                cn_dir / f"{country_tag}_all_neurons_random_layer{s}.json"
                for s in random_seeds
            ]
        else:
            entry = anchor_paths[loc]
            rand_files = [Path(entry["random"][s]) for s in random_seeds]
            slr_files = [Path(entry["slr"][s]) for s in random_seeds]
        for neuron_file in rand_files + slr_files:
            _mask_for_neuron_file(neuron_file, loc, country)

    if not is_anchor_cp:
        return None
    result: dict[str, dict | None] = {}
    for loc, country, culture_file in locale_anchor:
        if culture_file is None:
            result[loc] = None
            continue
        country_tag = country.replace(" ", "")
        result[loc] = {
            "country": country,
            "culture": culture_file,
            "random": {
                s: cn_dir / f"{country_tag}_all_neurons_random{s}.json"
                for s in random_seeds
            },
            "slr": {
                s: cn_dir / f"{country_tag}_all_neurons_random_layer{s}.json"
                for s in random_seeds
            },
        }
    return result


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_checkpoint_trace(args) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    setup_torch_optimizations()
    cfg = _load_config(Path(args.config))
    defaults = cfg.get("defaults", {})
    runs = cfg.get("runs", [])

    only_run = getattr(args, "run_id", None)
    if only_run:
        runs = [r for r in runs if r["id"] == only_run]
        if not runs:
            raise ValueError(f"run_id {only_run!r} not in {args.config}")

    force = bool(getattr(args, "force", False))

    for run in runs:
        locales = run["locales"]
        if run["source"]["kind"] == "hf_revisions":
            slug_name = run["source"]["repo"]
        else:
            base = Path(run["source"]["base_path"])
            slug_name = str(base / run["source"]["checkpoints"][0]["path"])
        slug = model_slug(slug_name)

        all_jobs = list(iter_checkpoints(run))
        if not all_jobs:
            continue
        anchor_job = all_jobs[-1]
        non_anchor_jobs = all_jobs[:-1]

        # Pass 1 — anchor cp: full pipeline
        LOGGER.info(
            "[%s] Pass 1 (anchor=%s): loading %s",
            run["id"],
            anchor_job.label,
            anchor_job.load_kwargs["model_name"],
        )
        load_kwargs = dict(anchor_job.load_kwargs)
        load_kwargs.setdefault("compile_model", False)
        model, tokenizer = load_model_and_tokenizer(
            **load_kwargs,
            dtype=defaults["dtype"],
        )
        try:
            anchor_paths = _run_neuron_pipeline_for_checkpoint(
                model=model,
                tokenizer=tokenizer,
                model_name=anchor_job.load_kwargs["model_name"],
                architecture=run["architecture"],
                revision_label=anchor_job.label,
                locales=locales,
                defaults=defaults,
                force=force,
                anchor_paths=None,
            )
        finally:
            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        LOGGER.info("[%s] Pass 1 complete; anchor neurons fixed", run["id"])

        # Pass 2 — earlier cps: trace anchor (no re-selection)
        for job in non_anchor_jobs:
            LOGGER.info(
                "[%s] Pass 2 (trace=%s): loading %s",
                run["id"],
                job.label,
                job.load_kwargs["model_name"],
            )
            load_kwargs = dict(job.load_kwargs)
            load_kwargs.setdefault("compile_model", False)
            model, tokenizer = load_model_and_tokenizer(
                **load_kwargs,
                dtype=defaults["dtype"],
            )
            try:
                _run_neuron_pipeline_for_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    model_name=job.load_kwargs["model_name"],
                    architecture=run["architecture"],
                    revision_label=job.label,
                    locales=locales,
                    defaults=defaults,
                    force=force,
                    anchor_paths=anchor_paths,
                )
            finally:
                del model, tokenizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        from diss.analysis.checkpoint_trace_summary import build_trajectory_summary

        out_csv = OUTPUT_ROOT / f"cp_summary_{run['id']}.csv"
        build_trajectory_summary(
            run=run, slug=slug, out_csv=out_csv, random_seeds=defaults["random_seeds"]
        )
        LOGGER.info("[%s] trajectory summary -> %s", run["id"], out_csv)
